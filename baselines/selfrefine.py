"""Self-Refine (Madaan et al., 2023) baseline on the SuperGPQA subset.

INIT -> (FEEDBACK -> REFINE)* loop, following src/gsm/run.py in madaan/self-refine: the loop
stops when the feedback says the answer is correct, or after --max-iters refinements. The whole
history stays in the conversation, as in the paper. INIT uses the official SuperGPQA zero-shot
prompt, so iteration 0 is directly comparable to the plain baseline.

Output lines carry the final answer in the same schema as generate.py, so score.py --k 1 scores
them unchanged; the per-iteration detail sits under "selfrefine".

Copied from /nas-ssd2/hwang/tianyi/supergpqa_eval/selfrefine.py and tailored to openai/gpt-oss-20b on
the 300-question dev split, with the same changes as generate.py (reasoning effort high, no
Qwen-specific sampling settings, max_tokens inside the 32,768-token window). The prompts and the
loop are unchanged. Added here: --k runs the whole loop k independent times per question (sample
indices 0..k-1, each with its own seeds), so score.py can report avg@k and pass@k as it does for
the direct baseline; index 0 is the same run a k=1 call would make.

--family qwen switches back to the original's Qwen request (thinking switch, top_k, min_p,
presence penalty) for the Qwen servers; the default, gptoss, is unchanged.

An HLE row (hle_format.py) is asked in HLE's own format (system prompt), with FEEDBACK and REFINE
prompts that do not mention options (hle_format.FEEDBACK_PROMPT, REFINE_PROMPT). A GPQA-Diamond or
MATH row gets its dataset's prompts (tasks.py): GPQA the multiple-choice FEEDBACK and a REFINE that
asks for a letter A-D, math HLE's FEEDBACK and a REFINE that asks for the answer in \\boxed{}.

--recover-feedback (off by default, so earlier runs resume unchanged): a FEEDBACK turn cut off at its
token limit before any visible text would leave REFINE an empty critique (and never stop the loop);
it is asked once, in the same way, for the critique its reasoning supports or 'it is correct'
(tasks.feedback_recover_prompts). Measured before it existed: gpt-oss at high effort left 14% of
its SuperGPQA feedback turns and 44% of its HLE ones empty at the 16,384-token limit.

The paper runs up to 4 FEEDBACK -> REFINE rounds (Section 3.1); --max-iters is 2 by default, as the
pipeline's in-executor self_refine_high (2 critic -> solver rounds), and run_baselines.py passes 2
as well since 2026-10-07 (4 before: its selfrefine_it4_* files).

--recover (off by default, so the loop above is unchanged without it): an answer turn (INIT or
REFINE) cut off at its token limit without an answer is asked once for the letter its reasoning
supports (recover.py, as the debate executor does), and that reply is the turn's answer for the
rest of the loop; if the last answer still names no letter, the most recent answer that did is
the final one (the executor's last-commit read). Each turn's record keeps the original reply
under "recovery"; "fell_back" marks a final answer taken from an earlier turn.
"""
import argparse
import asyncio
import json
import os
import random
import time

import aiohttp
from tqdm import tqdm

import tasks
from generate import question_messages, request_one, sample_seed
from recover import answer_of, needs_recovery, recover, with_recovery

# SuperGPQA's FEEDBACK and REFINE prompts (tasks.py has every dataset's)
FEEDBACK_PROMPT, REFINE_PROMPT = tasks.FEEDBACK_PROMPT, tasks.REFINE_PROMPT

STOP_PHRASE = tasks.STOP_PHRASE     # read through LaTeX and markdown: tasks.says_correct


def load_done(path):
    done = set()                                     # (question id, sample index) pairs
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("error") is None:
                    done.add((r["id"], r.get("sample_idx", 0)))
    return done


def model_settings(args) -> dict:
    """The request fields that depend on the model family. gpt-oss takes a reasoning effort;
    Qwen takes the thinking switch and the sampling settings of its model card (as in the
    collaborator's original script)."""
    if args.family == "qwen":
        return {"top_k": args.top_k, "min_p": args.min_p, "presence_penalty": args.presence_penalty,
                "chat_template_kwargs": {"enable_thinking": args.thinking}}
    return {"reasoning_effort": args.reasoning_effort}      # gpt-oss: low / medium / high


async def main(args):
    tasks.set_explain(args.explain)                  # the first prompt (see tasks.py)
    with open(args.data) as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    done = load_done(args.out)
    # run index 0 of every question first, so an interrupted run still has a complete first pass
    todo = []
    for s in range(args.k):
        wave = [(it, s) for it in items if (it["id"], s) not in done]
        random.Random(s).shuffle(wave)
        todo += wave
    n_done = sum(1 for it in items for s in range(args.k) if (it["id"], s) in done)
    print(f"{len(items)} questions x k={args.k}: {n_done} done, {len(todo)} to run (max_iters={args.max_iters})")
    if not todo:
        return

    endpoints = [e.rstrip("/") + "/v1/chat/completions" for e in args.endpoints.split(",")]
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    pbar = tqdm(total=len(todo), smoothing=0.05, dynamic_ncols=True)
    stats = {"tokens": 0, "errors": 0, "stopped": 0, "iters": 0, "recovered": 0, "t0": time.time()}
    out_f = open(args.out, "a")

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:

        async def ask(session, url, messages, seed, max_tokens):
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": seed,
                **model_settings(args),
            }
            data, err = await request_one(session, url, payload, args.max_retries)
            if data is None:
                return None, err
            choice = data["choices"][0]
            got = {
                "content": choice["message"].get("content"),
                "reasoning": choice["message"].get("reasoning") or choice["message"].get("reasoning_content"),
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": data["usage"]["completion_tokens"],
            }
            if "max_tokens_cut" in data:               # the conversation left less room than asked
                got["max_tokens_cut"] = data["max_tokens_cut"]
            return got, None

        async def run(job_idx, item, s):
            url = endpoints[job_idx % len(endpoints)]
            # runs of one question are 100000 apart, so their feedback/refine seeds never collide
            seed0 = (sample_seed(item["id"], 0) + 100000 * s) % (2**31)
            async with sem:
                # Only the visible answers go back into the history, never the thinking blocks.
                messages = question_messages(item)
                turns, err = [], None

                async def answer(seed, k):
                    """An answer turn (k = 0 for INIT, i + 1 for REFINE i) on the conversation so far;
                    with --recover, a reply cut off without an answer is recovered."""
                    got, e = await ask(session, url, messages, seed, args.max_tokens)
                    if e or not (args.recover and needs_recovery(got, item)):
                        return got, e
                    fix, e = await recover(session, url, args, messages, got, item, seed0 + 3000 + 10 * k)
                    if e:
                        return None, e
                    stats["recovered"] += fix["calls"] > 0
                    return with_recovery(got, fix), None

                step, e = await answer(seed0, 0)
                if e:
                    err = e
                else:
                    turns.append({"role": "init", **step})
                    messages.append({"role": "assistant", "content": step["content"] or ""})
                    for it in range(args.max_iters):
                        messages.append({"role": "user", "content": tasks.feedback_prompt(item)})
                        fb, e = await ask(session, url, messages, seed0 + 1000 + it, args.feedback_max_tokens)
                        if not e and args.recover_feedback and fb["finish_reason"] == "length" \
                                and not (fb["content"] or "").strip():
                            fix, e = await recover(session, url, args, messages, fb, item, seed0 + 4000 + 10 * it,
                                                   prompts=tasks.feedback_recover_prompts(item),
                                                   accept=lambda text: bool((text or "").strip()))
                            if not e:
                                stats["recovered"] += fix["calls"] > 0
                                fb = with_recovery(fb, fix)
                        if e:
                            err = e
                            break
                        turns.append({"role": "feedback", **fb})
                        messages.append({"role": "assistant", "content": fb["content"] or ""})
                        if tasks.says_correct(fb["content"]):
                            break
                        messages.append({"role": "user", "content": tasks.refine_prompt(item)})
                        rf, e = await answer(seed0 + 2000 + it, it + 1)
                        if e:
                            err = e
                            break
                        turns.append({"role": "refine", **rf})
                        messages.append({"role": "assistant", "content": rf["content"] or ""})

            answers = [t for t in turns if t["role"] in ("init", "refine")]
            final = answers[-1] if answers else None
            fell_back = False
            if args.recover and final is not None and answer_of(final["content"], item) is None:
                # the executor's last-commit read: the most recent answer that named a letter
                named = [t for t in answers[:-1] if answer_of(t["content"], item) is not None]
                if named:
                    final, fell_back = named[-1], True
            n_refine = sum(t["role"] == "refine" for t in turns)
            rec = {
                "id": item["id"],
                "sample_idx": s,
                "error": err,
                # score.py reads these three; they describe the final answer of the loop
                "content": final["content"] if final else None,
                "finish_reason": answers[-1]["finish_reason"] if answers else None,
                "completion_tokens": sum(t["completion_tokens"] for t in turns),
                "selfrefine": {
                    "n_refinements": n_refine,
                    "stopped_early": n_refine < args.max_iters and err is None,
                    "init_answer": answers[0]["content"] if answers else None,
                    "recovered_turns": sum(t.get("recovery", {}).get("calls", 0) > 0 for t in turns),
                    "fell_back": fell_back,
                    "turns": turns if args.save_turns else
                             [{k: v for k, v in t.items() if k != "reasoning"} for t in turns],
                },
            }
            if err:
                stats["errors"] += 1
            else:
                stats["stopped"] += rec["selfrefine"]["stopped_early"]
                stats["iters"] += n_refine
            stats["tokens"] += rec["completion_tokens"]

            async with lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                pbar.update(1)
                el = time.time() - stats["t0"]
                n = max(pbar.n, 1)
                pbar.set_postfix(tok_s=f"{stats['tokens'] / el:.0f}", refines=f"{stats['iters'] / n:.2f}",
                                 early=stats["stopped"], recovered=stats["recovered"], err=stats["errors"])

        await asyncio.gather(*(run(i, it, s) for i, (it, s) in enumerate(todo)))

    pbar.close()
    out_f.close()
    el = time.time() - stats["t0"]
    print(f"finished {len(todo)} runs in {el / 3600:.2f}h, {stats['tokens'] / el:.0f} tok/s, "
          f"avg refinements {stats['iters'] / max(len(todo), 1):.2f}, stopped early {stats['stopped']}, "
          f"errors {stats['errors']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--endpoints", default="http://localhost:7472")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--k", type=int, default=3, help="independent Self-Refine runs per question")
    p.add_argument("--max-iters", type=int, default=2, help="maximum FEEDBACK -> REFINE rounds after the initial answer")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=128)
    # prompt + max_tokens must fit the server's 32,768-token window; later turns carry the visible
    # answers and critiques of earlier turns (never their thinking), so they get more headroom
    p.add_argument("--max-tokens", type=int, default=24576)
    p.add_argument("--feedback-max-tokens", type=int, default=16384)
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
    p.add_argument("--save-turns", action="store_true", help="also keep the thinking of every turn")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--recover", action="store_true",
                   help="recover answer turns cut off at the token limit (see the module docstring)")
    p.add_argument("--recover-feedback", action="store_true",
                   help="also recover a feedback turn cut off at the token limit before any visible text: "
                        "one cheap call asks for the critique its reasoning supports (off by default)")
    asyncio.run(main(p.parse_args()))

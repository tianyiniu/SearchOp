"""Multi-agent debate (Du et al., ICML 2024, "Improving Factuality and Reasoning in Language Models
through Multiagent Debate") as an external baseline.

Follows the authors' code (composable-models/llm_multiagent_debate: mmlu/gen_mmlu.py, gsm/gen_gsm.py
and their eval_*.py):
  - 3 agents and 2 rounds, the paper's main setting ("three agents with two rounds of debate";
    rounds = 2 in the code is a first answer and one debate round). --agents, --rounds.
  - Round 1: each agent answers the question alone, with the same prompt as the direct baseline
    (the paper uses "the identical starting prompt" for its baselines and the debate).
  - Each later round: every agent's own conversation gets one user turn with the other agents'
    replies of the round before ("These are the solutions to the problem from other agents: ...
    One agent solution: ```...```") and the request for an updated answer (the MMLU wording for
    multiple choice and HLE, the GSM wording for math: tasks.debate_message). Its own earlier turns
    stay in its conversation. Every agent sees the round before, so a round's turns run in parallel.
  - The final answer: the most common answer among the agents' last replies; a tie goes to the
    lowest-numbered agent, and a reply with no answer does not vote (most_frequent and
    compute_accuracy in eval_mmlu.py; tasks.vote, which for math counts answers that math-verify
    finds equal as one). The scorer does this vote from "finals".
Changes, for these models and datasets:
  - The answer format is each dataset's own (the line 'Answer: X', \\boxed{} for math, HLE's
    format) in place of '(X)' or a \\boxed number, so every baseline is read by the same rule.
  - Only the visible reply goes into the conversations, never the thinking (as in selfrefine.py).
  - --recover (off by default): a reply cut off at its token limit with no answer is asked once
    for the answer its reasoning supports (recover.py), and that reply is what the other agents see.
  - --k runs the whole debate k independent times per question (sample indices 0..k-1).
  - A turn whose conversation leaves no room for a reply in the window (generate.no_room) keeps the
    agent's reply of the round before (in round 1: no reply, which does not vote), so the debate
    goes on and the question gets a result (2026-10-08; before, the run was an error and stayed
    missing); such a turn has finish_reason "no_room". A recovery with no room leaves the reply as
    it was cut off (recover.py).

Output: one line per (question, run): "finals" (the agents' last replies), the run's total
completion tokens (every turn, recovery included), and every turn under "mad" (without the thinking
unless --save-turns). Resumable: finished runs are skipped, and a failed request writes an error
line that the next call runs again.
"""
import argparse
import asyncio
import json
import random
import time

import aiohttp
from tqdm import tqdm

import tasks
from generate import no_room, request_one, sample_seed
from recover import add_request_args, needs_recovery, recover, with_recovery
from selfrefine import load_done, model_settings

SEED_OFFSET = 7_000_000        # far from the direct samples' (index < 1000), Self-Refine's and recovery's


def kept_reply(last: dict | None) -> dict:
    """The turn of an agent whose conversation left no room for a reply: its reply of the round
    before again (none in round 1), with no tokens of its own."""
    return {"content": last["content"] if last else None, "finish_reason": "no_room", "completion_tokens": 0}


async def main(args):
    tasks.set_explain(args.explain)                  # the first prompt (see tasks.py)
    with open(args.data) as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    done = load_done(args.out)
    todo = []                                        # every question's run 0 first, then run 1, ...
    for s in range(args.k):
        wave = [(it, s) for it in items if (it["id"], s) not in done]
        random.Random(s).shuffle(wave)
        todo += wave
    n_done = sum(1 for it in items for s in range(args.k) if (it["id"], s) in done)
    print(f"{len(items)} questions x k={args.k}: {n_done} done, {len(todo)} to run "
          f"({args.agents} agents, {args.rounds} rounds)")
    if not todo:
        return

    endpoints = [e.rstrip("/") + "/v1/chat/completions" for e in args.endpoints.split(",")]
    # a debate has up to `agents` requests in flight, so at most concurrency // agents debates run
    # at once and they finish in the order above
    runs_at_once = asyncio.Semaphore(max(1, args.concurrency // args.agents))
    lock = asyncio.Lock()
    pbar = tqdm(total=len(todo), smoothing=0.05, dynamic_ncols=True)
    stats = {"tokens": 0, "errors": 0, "recovered": 0, "unanimous": 0, "no_room": 0, "t0": time.time()}
    out_f = open(args.out, "a")

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:

        async def turn(url, messages, seed, max_tokens, item):
            """One agent's reply to its conversation; with --recover, a reply cut off before its
            answer is recovered. Returns (reply, None) or (None, error)."""
            payload = {"model": args.model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": args.temperature, "top_p": args.top_p, "seed": seed,
                       **model_settings(args)}
            data, err = await request_one(session, url, payload, args.max_retries)
            if data is None:
                return None, err
            choice = data["choices"][0]
            got = {"content": choice["message"].get("content"),
                   "reasoning": choice["message"].get("reasoning") or choice["message"].get("reasoning_content"),
                   "finish_reason": choice.get("finish_reason"),
                   "completion_tokens": data["usage"]["completion_tokens"]}
            if "max_tokens_cut" in data:               # the conversation left less room than asked
                got["max_tokens_cut"] = data["max_tokens_cut"]
            if not (args.recover and needs_recovery(got, item)):
                return got, None
            fix, err = await recover(session, url, args, messages, got, item, seed + 500)
            if err:
                return None, err
            stats["recovered"] += fix["calls"] > 0
            return with_recovery(got, fix), None

        async def run(job_idx, item, s):
            url = endpoints[job_idx % len(endpoints)]
            base = sample_seed(item["id"], 0) + SEED_OFFSET + 1000 * s
            contexts = [tasks.messages(item) for _ in range(args.agents)]     # each agent's conversation
            rounds, err = [], None
            async with runs_at_once:
                for t in range(args.rounds):
                    if t > 0:
                        for i, ctx in enumerate(contexts):
                            others = [r["content"] or "" for j, r in enumerate(rounds[-1]) if j != i]
                            ctx.append({"role": "user", "content": tasks.debate_message(item, others)})
                    limit = args.max_tokens if t == 0 else args.debate_max_tokens
                    got = await asyncio.gather(*(turn(url, ctx, (base + 10 * t + i) % (2**31), limit, item)
                                                 for i, ctx in enumerate(contexts)))
                    stats["no_room"] += sum(no_room(e) for _, e in got)
                    got = [(kept_reply(rounds[-1][i] if rounds else None), None) if no_room(e) else (g, e)
                           for i, (g, e) in enumerate(got)]
                    err = next((e for _, e in got if e), None)
                    if err:
                        break
                    replies = [g for g, _ in got]
                    for ctx, g in zip(contexts, replies):
                        ctx.append({"role": "assistant", "content": g["content"] or ""})
                    rounds.append(replies)

            tokens = sum(r["completion_tokens"] for rs in rounds for r in rs)
            finals = [r["content"] for r in rounds[-1]] if not err else None
            rec = {
                "id": item["id"],
                "sample_idx": s,
                "error": err,
                "finals": finals,                    # the scorer votes over the answers of these
                "finish_reason": [r["finish_reason"] for r in rounds[-1]] if finals else None,
                "completion_tokens": tokens,
                "mad": {"agents": args.agents, "rounds": args.rounds,
                        "recovered_turns": sum(r.get("recovery", {}).get("calls", 0) > 0
                                               for rs in rounds for r in rs),
                        "turns": [[r if args.save_turns else {k: v for k, v in r.items() if k != "reasoning"}
                                   for r in rs] for rs in rounds]},
            }
            if err:
                stats["errors"] += 1
            else:
                answers = [tasks.answer_of(c, item) for c in finals]
                stats["unanimous"] += len(set(answers)) == 1 and answers[0] is not None
            stats["tokens"] += tokens
            async with lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                pbar.update(1)
                el = time.time() - stats["t0"]
                pbar.set_postfix(tok_s=f"{stats['tokens'] / el:.0f}", unanimous=stats["unanimous"],
                                 recovered=stats["recovered"], err=stats["errors"])

        await asyncio.gather(*(run(i, it, s) for i, (it, s) in enumerate(todo)))

    pbar.close()
    out_f.close()
    el = time.time() - stats["t0"]
    print(f"finished {len(todo)} debates in {el / 3600:.2f}h, {stats['tokens'] / el:.0f} tok/s, "
          f"agents agree at the end in {stats['unanimous']}, recovered turns {stats['recovered']}, "
          f"turns with no room {stats['no_room']}, errors {stats['errors']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=3, help="independent debates per question")
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--rounds", type=int, default=2, help="the first answer counts as round 1")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=96, help="requests in flight (debates: this / agents)")
    # prompt + max_tokens must fit the server's 32,768-token window: the first round has the
    # direct baseline's room; a debate turn carries the visible replies of the round before
    # (never their thinking), so it gets Self-Refine's
    p.add_argument("--max-tokens", type=int, default=28672)
    p.add_argument("--debate-max-tokens", type=int, default=24576)
    p.add_argument("--no-thinking", dest="thinking", action="store_false", help="qwen only")
    p.add_argument("--save-turns", action="store_true", help="also keep the thinking of every turn")
    p.add_argument("--recover", action="store_true",
                   help="recover replies cut off at the token limit (see the module docstring)")
    add_request_args(p)
    asyncio.run(main(p.parse_args()))

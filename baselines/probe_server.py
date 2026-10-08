#!/usr/bin/env python3
"""Check a model server before a long baseline run (run_baselines.py): the model is served with the
window the token limits assume, and every kind of request the baselines send works on it, on every
dataset, and gives an answer that can be read. Every request and reply is saved for review.

    python3 baselines/probe_server.py --model gptoss-20b                # the model's default port
    python3 baselines/probe_server.py --model qwen35-4b --port 7473 --e2e

--model is one of run_baselines.py's models, with the same settings. Checks, each PASS or FAIL:
  1. server     the port serves the model, with a window of at least 32,768 tokens
  2. first      the first request of one question of each dataset (SuperGPQA, GPQA-Diamond, MATH,
                and HLE twice: a multiple-choice and an open-answer question), in parallel, with
                the real token limit: it finishes (or is cut off at the limit) and its answer can
                be read; its thinking and visible text are measured
  3. on each of those replies, in parallel, the three later turns the baselines send:
       recovery   the cheap follow-up for a reply cut off before its answer (Qwen with thinking
                  off, gpt-oss at low effort; recover.py), on a made-up cut-off: a readable answer
       feedback   Self-Refine's feedback turn (24,576 tokens, as run_baselines.py sends it): it gives
                  some text; one cut off before any text is recovered as selfrefine.py
                  --recover-feedback does, and passes if that gives text
       debate     a MAD debate turn, the reply shown as both other agents': a readable answer
With --e2e it then runs run_baselines.py on the first question of SuperGPQA, GPQA-Diamond and
MATH, one run of each method (about 45 model calls). HLE is left out there: grading it calls the
paid judge model.

Everything goes to baselines/results/probes/<model>_<date>_<time>/ (--out): probe.log (what is
printed), probe.json (every request: its messages and settings; every reply: visible text,
thinking, finish reason, tokens; and each check's verdict), and e2e/ (run_baselines.py's files).

The feedback call and the other debate agents see only a reply's visible text, never its thinking.
A model that keeps its reasoning hidden gives them little to read (gpt-oss on SuperGPQA: median
9 characters, 'Answer: C'), so a visible reply under 200 characters is flagged (NOTE, not a FAIL).
Exit status 0 when every check passes.
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_baselines import MODELS, VENV_PYTHON, WINDOW, use_python  # noqa: E402  (no imports outside the stdlib)

# (name, file, which row): the first question, or the first of an HLE answer type
QUESTIONS = (("supergpqa", "datasets/supergpqa_2k_test.json", None),
             ("gpqa", "datasets/gpqa_diamond_test.json", None),
             ("math", "datasets/math_l5_test.json", None),
             ("hle-mc", "datasets/hle_text_test_200.json", "multipleChoice"),
             ("hle-open", "datasets/hle_text_test_200.json", "exactMatch"))
E2E = ("datasets/supergpqa_2k_test.json", "datasets/gpqa_diamond_test.json", "datasets/math_l5_test.json")
SHORT_VISIBLE = 200
results = []                      # (check, ok, detail)
records = []                      # every request and reply, for probe.json


class Tee:
    """Print to the terminal and to probe.log."""

    def __init__(self, path: Path):
        self.file, self.out = open(path, "a"), sys.stdout

    def write(self, text):
        self.out.write(text)
        self.file.write(text)

    def flush(self):
        self.out.flush()
        self.file.flush()


def report(check: str, ok: bool, detail: str) -> None:
    results.append((check, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {check:<20} {detail}", flush=True)


def note(text: str) -> None:
    print(f"NOTE  {'':<20} {text}", flush=True)


def check_server(port: int, name: str) -> bool:
    try:
        served = json.load(urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=10))
    except OSError as err:
        report("server", False, f"no server on port {port} ({err})")
        return False
    found = {m.get("id"): m.get("max_model_len") for m in served.get("data", [])}
    if name not in found:
        report("server", False, f"port {port} serves {list(found)}, not {name}")
        return False
    window = found[name]
    report("server", window is None or window >= WINDOW, f"{name} on port {port}, window {window} tokens")
    return window is None or window >= WINDOW


def pick(path: str, answer_type: str | None) -> dict:
    rows = json.loads((ROOT / path).read_text())
    return rows[0] if answer_type is None else next(r for r in rows if r.get("answer_type") == answer_type)


async def probe(args, ns) -> None:
    import aiohttp

    import tasks
    tasks.set_explain(MODELS[args.model].get("explain", False))      # the first prompt, as the run sends it
    from generate import request_one
    from recover import recover
    from selfrefine import model_settings

    url = f"http://localhost:{args.port}/v1/chat/completions"
    rows = {name: pick(path, t) for name, path, t in QUESTIONS if (ROOT / path).exists()}
    for name, path, _ in QUESTIONS:
        if name not in rows:
            print(f"skip  {name}: {path} is missing")

    async def ask(session, check, item, messages, max_tokens, seed):
        payload = {"model": ns.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": ns.temperature, "top_p": ns.top_p, "seed": seed, **model_settings(ns)}
        t0 = time.time()
        data, err = await request_one(session, url, payload, ns.max_retries)
        got = None
        if data is not None:
            choice = data["choices"][0]
            got = {"content": choice["message"].get("content"),
                   "reasoning": choice["message"].get("reasoning") or choice["message"].get("reasoning_content"),
                   "finish_reason": choice.get("finish_reason"),
                   "completion_tokens": data["usage"]["completion_tokens"], "seconds": round(time.time() - t0, 1)}
        records.append({"check": check, "question_id": item["id"], "request": payload, "reply": got, "error": err,
                        "answer_read": tasks.answer_of(got["content"], item) if got else None})
        return got, err

    def describe(got, item):
        answer = tasks.answer_of(got["content"], item)
        return answer, (f"finish {got['finish_reason']}, {got['completion_tokens']} tokens, "
                        f"{got['seconds']:.0f}s, thinking {len(got['reasoning'] or '')} chars, "
                        f"visible {len(got['content'] or '')} chars, answer {answer!r}")

    async def later_turns(session, name, item, got):
        """Recovery, feedback and debate on one first reply."""
        reply = got["content"] or ""
        base = tasks.messages(item) + [{"role": "assistant", "content": reply}]
        fb_messages = base + [{"role": "user", "content": tasks.feedback_prompt(item)}]
        cut = {"content": "", "finish_reason": "length",
               "reasoning": (got["reasoning"] or "I work through the question step by step.")[:4000]}
        rec, fb, deb = await asyncio.gather(
            recover(session, url, ns, tasks.messages(item), cut, item, 99),
            ask(session, f"feedback: {name}", item, fb_messages, 24576, 77),
            ask(session, f"debate: {name}", item,
                base + [{"role": "user", "content": tasks.debate_message(item, [reply, reply])}], 24576, 88))
        fix, err = rec
        shown = tasks.messages(item) + [{"role": "assistant", "content": "(the made-up cut-off reply's thinking)"}]
        records.append({"check": f"recovery: {name}", "question_id": item["id"],
                        "request": {"messages": shown, "prompts": list(tasks.recover_prompts(item))},
                        "reply": fix, "error": err,
                        "answer_read": tasks.answer_of(fix["content"], item) if fix else None})
        if err:
            report(f"recovery: {name}", False, err)
        else:
            answer = tasks.answer_of(fix["content"], item)
            report(f"recovery: {name}", answer is not None,
                   f"{fix['calls']} call(s), {fix['completion_tokens']} tokens, answer {answer!r}")
        got, err = fb
        if not err and got["finish_reason"] == "length" and not (got["content"] or "").strip():
            # as selfrefine.py --recover-feedback does: ask once for the critique
            fix, err = await recover(session, url, ns, fb_messages, got, item, 66,
                                     prompts=tasks.feedback_recover_prompts(item),
                                     accept=lambda text: bool((text or "").strip()))
            records.append({"check": f"feedback recovery: {name}", "question_id": item["id"],
                            "request": {"messages": fb_messages, "prompts": list(tasks.feedback_recover_prompts(item))},
                            "reply": fix, "error": err, "answer_read": None})
            if not err:
                note(f"feedback on {name} cut off before any text; recovered in {fix['calls']} call(s): "
                     f"{(fix['content'] or '')[-100:]!r}")
                got = {**got, "content": fix["content"], "recovered": True}
        if err:
            report(f"feedback: {name}", False, err)
        else:
            _, text = describe(got, item)
            report(f"feedback: {name}", bool((got["content"] or "").strip()),
                   f"{text.split(', answer')[0]}, says it is correct: {tasks.says_correct(got['content'])}"
                   + (" (recovered)" if got.get("recovered") else ""))
        got, err = deb
        if err:
            report(f"debate: {name}", False, err)
        else:
            answer, text = describe(got, item)
            report(f"debate: {name}", got["finish_reason"] == "stop" and answer is not None, text)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        print(f"\nfirst requests ({', '.join(rows)}), up to {args.max_tokens} tokens each; this can take minutes",
              flush=True)
        firsts = await asyncio.gather(*(ask(session, f"first: {name}", row, tasks.messages(row), args.max_tokens, 1234)
                                        for name, row in rows.items()))
        first = {}
        for (name, row), (got, err) in zip(rows.items(), firsts):
            if err:
                report(f"first: {name}", False, err)
                continue
            answer, text = describe(got, row)
            key = row.get("answer_letter", row.get("answer"))
            report(f"first: {name}", answer is not None or got["finish_reason"] == "length",
                   f"{text} (key {str(key)[:40]!r})")
            if got["finish_reason"] == "length":
                note("cut off at the token limit; with recovery the baselines ask it for the answer")
            if len(got["content"] or "") < SHORT_VISIBLE:
                note(f"visible reply under {SHORT_VISIBLE} characters: Self-Refine's feedback and the other MAD "
                     "agents would see little reasoning")
            first[name] = got

        print("\nlater turns (recovery, feedback, debate) on each first reply", flush=True)
        await asyncio.gather(*(later_turns(session, name, rows[name], got) for name, got in first.items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--port", type=int, default=None, help="the server's port on localhost (default: the model's)")
    p.add_argument("--e2e", action="store_true",
                   help="also run every method once on 1 SuperGPQA, 1 GPQA and 1 MATH question")
    p.add_argument("--max-tokens", type=int, default=28672, help="first requests (the baselines' limit)")
    p.add_argument("--out", default=None, help="where the record goes (default baselines/results/probes/...)")
    p.add_argument("--python", default=VENV_PYTHON, help="the project's venv")
    args = p.parse_args()
    use_python(args.python)
    m = MODELS[args.model]
    if args.port is None:
        args.port = m["port"]
    out = Path(args.out or HERE / "results/probes" / f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "probe.log")
    # the request settings, as the baseline scripts read them
    ns = argparse.Namespace(model=m["name"], temperature=m["temperature"], top_p=m["top_p"], family=m["family"],
                            top_k=m["top_k"], min_p=m["min_p"], presence_penalty=m["presence_penalty"],
                            reasoning_effort=m["reasoning_effort"], thinking=True, max_retries=2)
    print(f"probe: {args.model} = {m['name']} on port {args.port} ({m['family']} settings); record in {out}")
    if check_server(args.port, m["name"]):
        asyncio.run(probe(args, ns))
        if args.e2e:
            (out / "e2e").mkdir(exist_ok=True)
            for data in E2E:
                print(f"\nend to end: every method once on the first question of {data}", flush=True)
                res = subprocess.run([sys.executable, str(HERE / "run_baselines.py"), "--model", args.model,
                                      "--port", str(args.port), "--data", str(ROOT / data), "--limit", "1",
                                      "--k", "1", "--results-dir", str(out / "e2e")], capture_output=True, text=True)
                (out / "e2e" / f"{Path(data).stem}.log").write_text(res.stdout + res.stderr)
                tail = res.stdout[res.stdout.find("## "):] if "## " in res.stdout else res.stdout[-2000:]
                print(tail if res.returncode == 0 else res.stdout[-2000:] + res.stderr[-2000:])
                report(f"end to end: {Path(data).stem}", res.returncode == 0,
                       "all four methods ran" if res.returncode == 0 else f"exit {res.returncode}")
    (out / "probe.json").write_text(json.dumps({"model": args.model, "settings": vars(ns), "port": args.port,
                                                "checks": [{"check": c, "ok": ok, "detail": d} for c, ok, d in results],
                                                "requests": records}, indent=1, ensure_ascii=False))
    failed = [c for c, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed"
          + (f"; failed: {', '.join(failed)}" if failed else "") + f"\nrecord: {out}")
    sys.exit(1 if failed or not results else 0)


if __name__ == "__main__":
    main()

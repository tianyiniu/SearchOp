#!/usr/bin/env python3
"""Run the external baselines on one dataset, one method after another, then score them and print
one table.

    python3 baselines/run_baselines.py --model qwen35-9b --data datasets/gpqa_diamond_test.json

--model names one of the models below; it sets the served model, the sampling settings (each
model card's thinking-mode settings, as the pipeline uses them) and the name in the result files.
The server is http://localhost:<--port>, by default the port its deploy script uses
(Model_hosting/deploy_*.sh). The script runs in the project's venv (--python): started with any
other Python (python3), it starts itself again under that one.

    model        served as                temperature, top_p, other     port   file tag
    qwen35-9b    Qwen/Qwen3.5-9B          1.0, 0.95, top_k 20,          7472   qwen35_9b_think
                                          presence penalty 1.5, thinking on
    qwen35-4b    Qwen/Qwen3.5-4B          the same                      7473   qwen35_4b_think
    gptoss-20b   openai/gpt-oss-20b       1.0, 1.0, reasoning effort    7472   gptoss20b_high_explain
                                          high; the first prompt asks for
                                          the reasoning in the reply
                                          (tasks.EXPLAIN_SENTENCE; not HLE)

Methods (--methods; the table keeps this order):
    direct      direct chain-of-thought: one reply to the dataset's prompt (generate.py)
    sc          self-consistency (Wang et al.): the most common answer of --sc-n direct replies
                (generate.py's samples, shared with direct; self_consistency.py)
    selfrefine  Self-Refine (Madaan et al.): up to --sr-iters feedback -> refine rounds (selfrefine.py);
                2 by default since 2026-10-07, as the pipeline's self_refine_high (2 critic -> solver
                rounds) and its external Self-Refine; the paper's 4 with --sr-iters 4 (the
                selfrefine_it4_* files were made so)
    mad         multi-agent debate (Du et al.): --mad-agents agents, --mad-rounds rounds, majority
                vote of the last round (mad.py)
--data all runs every method on the test splits of all four datasets, one dataset after another:
SuperGPQA (supergpqa_2k_test, 1,000 questions; it holds the 300 of supergpqa_600_test, and no
question of either train split), HLE (hle_text_test_200, 200), GPQA-Diamond (198) and MATH level 5
(math_l5_test, 662). Each dataset gets its table, and all four tables also go
into table_<tag>_all[_rec].md. A dataset that cannot finish is reported, and the others still run.

Each method runs --k times per question (default 3), each run with its own seeds. Every method
starts from the same prompt, the dataset's own (tasks.py). With recovery (on by default;
--no-recover for the methods as published) a reply cut off at its token limit before its answer
is asked once for the answer its reasoning supports (recover.py), for every method alike; a
Self-Refine feedback turn cut off before any text is asked for its critique the same way.

Files: baselines/results/<method>_<tag>_<dataset>[_rec].jsonl; the direct file is the one the
pipeline scripts use, so samples made there are reused. probe_server.py checks a server with
these settings before a long run. The table: baselines/results/
table_<tag>_<dataset>[_rec].md, with the per-question answers in the .json beside it.

Metrics, on the questions every listed method has finished (in %):
    avg@1     run 1 alone
    vote@3    the most common answer of the 3 runs (a tie goes to the earliest run)
    avg@3     the mean of the 3 runs
    pass@3    right in at least one of the 3 runs
    tokens / run        completion tokens of one run (every call, recovery included), mean
    tokens / question   the same, summed over the 3 runs of a question, mean over the questions
Grading: letters against the key; MATH by math-verify; HLE by the judge model (judge_answers.py:
gpt-6-luna, a paid API, key OPENAI_API_KEY in .env; every verdict is cached and shared with the
pipeline, so an answer is graded once). Every step resumes (finished samples are kept), so after a stop, run the same command
again; a method is tried up to --attempts times while requests fail. --score-only makes no model
calls.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
tasks = None                      # imported in main, once the script runs in the venv

METHODS = ("direct", "sc", "selfrefine", "mad")
TEST_SETS = ("datasets/supergpqa_2k_test.json", "datasets/hle_text_test_200.json",   # --data all, in this order
             "datasets/gpqa_diamond_test.json", "datasets/math_l5_test.json")
VENV_PYTHON = "/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python"
WINDOW = 32768                    # the deploy scripts' --max-model-len; the scripts' token limits assume it
QWEN_THINKING = {"family": "qwen", "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                 "presence_penalty": 1.5, "reasoning_effort": "high"}
MODELS = {
    "qwen35-9b": {"name": "Qwen/Qwen3.5-9B", "tag": "qwen35_9b_think", "port": 7472, **QWEN_THINKING},
    "qwen35-4b": {"name": "Qwen/Qwen3.5-4B", "tag": "qwen35_4b_think", "port": 7473, **QWEN_THINKING},
    # gpt-oss keeps its reasoning hidden, so its first prompt asks for it in the reply (tasks.EXPLAIN);
    # the tag says so, and its files never mix with the pipeline's gptoss20b_high ones
    "gptoss-20b": {"name": "openai/gpt-oss-20b", "tag": "gptoss20b_high_explain", "port": 7472, "family": "gptoss",
                   "temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                   "reasoning_effort": "high", "explain": True},   # top_k, min_p, presence penalty: not sent
}


def use_python(python: str) -> None:
    """Start this script again under `python` (the project's venv) unless it already runs there."""
    if Path(sys.prefix).resolve() == Path(python).parent.parent.resolve():
        return
    if not Path(python).exists():
        raise SystemExit(f"{python} does not exist: pass --python with the project's venv")
    os.execv(python, [python, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])


def endpoint(args) -> str:
    return f"http://localhost:{args.port}"


def label(method: str, args) -> str:
    return {"direct": "Direct CoT",
            "sc": f"Self-consistency ({args.sc_n} samples)",
            "selfrefine": f"Self-Refine (up to {args.sr_iters} rounds)",
            "mad": f"MAD ({args.mad_agents} agents, {args.mad_rounds} rounds)"}[method]


def files(args) -> dict:
    """method -> the file the table reads; "direct_raw": generate.py's own output."""
    rd, name, rec = Path(args.results_dir), f"{args.tag}_{Path(args.data).stem}", "_rec" if args.recover else ""
    return {"direct_raw": rd / f"direct_{name}.jsonl",
            "direct": rd / f"direct_{name}{rec}.jsonl",
            "sc": rd / f"sc{args.sc_n}_{name}{rec}.jsonl",
            "selfrefine": rd / f"selfrefine_it{args.sr_iters}_{name}{rec}.jsonl",
            "mad": rd / f"mad_a{args.mad_agents}r{args.mad_rounds}_{name}{rec}.jsonl"}


def load_runs(path: Path, ids: set, k: int) -> dict:
    """(question id, run index < k) -> record, error-free ones only; the last write wins."""
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:              # partial last line from a killed run
                continue
            if r.get("error") is None and r.get("id") in ids and r.get("sample_idx", k) < k:
                out[(r["id"], r["sample_idx"])] = r
    return out


def missing(path: Path, ids: set, k: int) -> int:
    return len(ids) * k - len(load_runs(path, ids, k))


# --- running -------------------------------------------------------------------------------------

def request_args(args) -> list[str]:
    """The model's request settings, as generate.py, recover.py, selfrefine.py and mad.py spell them."""
    m = MODELS[args.model]
    return ["--endpoints", endpoint(args), "--model", m["name"], "--temperature", str(m["temperature"]),
            "--top-p", str(m["top_p"]), "--family", m["family"], "--top-k", str(m["top_k"]),
            "--min-p", str(m["min_p"]), "--presence-penalty", str(m["presence_penalty"]),
            "--reasoning-effort", m["reasoning_effort"], "--max-retries", str(args.max_retries),
            *(["--explain"] if m.get("explain") else [])]


def call(script: str, *argv) -> None:
    cmd = [sys.executable, str(HERE / script), *map(str, argv)]
    print("$ " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


class ServerDown(Exception):
    """The model server stopped answering, so every later request would fail too: the run stops."""


def server_up(args) -> None:
    """Raise ServerDown unless the port answers. (check_server, at the start, also checks the model.)"""
    try:
        urllib.request.urlopen(endpoint(args) + "/v1/models", timeout=10).read()
    except OSError as err:
        raise ServerDown(f"the server on port {args.port} stopped answering ({err}). Finished work is kept: "
                         f"restart the server and run the same command again") from None


def repeat(what: str, left, attempt_fn, args) -> None:
    """Run attempt_fn until left() is 0, up to --attempts times; stop if it is still not 0. Before
    each attempt, and when samples are still missing, the server must answer (else ServerDown)."""
    for attempt in range(1, args.attempts + 1):
        n = left()
        if n == 0:
            return
        server_up(args)
        print(f"\n=== {what}: attempt {attempt}, {n} samples to run", flush=True)
        attempt_fn()
    if (n := left()) != 0:
        server_up(args)
        raise SystemExit(f"{what}: {n} samples still missing after {args.attempts} attempts "
                         f"(failed requests; run the same command again)")


def run_direct(args, f: dict, ids: set, k: int) -> None:
    """generate.py up to k samples per question, then recover.py's copy (with recovery)."""
    limit = ["--limit", args.limit] if args.limit else []
    common = ["--data", args.data, "--concurrency", args.concurrency]

    def left():                       # replies to make, then (with recovery) replies to copy or recover
        n = missing(f["direct_raw"], ids, k)
        return n if n or not args.recover else missing(f["direct"], ids, k)

    def attempt():
        call("generate.py", *common, "--out", f["direct_raw"], "--k", k, "--max-tokens", args.direct_max_tokens,
             *limit, *request_args(args))
        if args.recover:
            call("recover.py", *common, "--results", f["direct_raw"], "--out", f["direct"], *request_args(args))
    repeat(f"direct samples (k={k})", left, attempt, args)


def run_method(method: str, args, f: dict, ids: set) -> None:
    limit = ["--limit", args.limit] if args.limit else []
    common = ["--data", args.data, "--concurrency", args.concurrency, *limit]
    recover = ["--recover"] if args.recover else []
    if method == "direct":
        run_direct(args, f, ids, args.k)
    elif method == "sc":
        run_direct(args, f, ids, args.k * args.sc_n)
        call("self_consistency.py", "--data", args.data, "--samples", f["direct"], "--n", args.sc_n,
             "--runs", args.k, "--out", f["sc"], *limit)
    elif method == "selfrefine":
        repeat("Self-Refine", lambda: missing(f["selfrefine"], ids, args.k),
               lambda: call("selfrefine.py", *common, "--out", f["selfrefine"], "--k", args.k,
                            "--max-iters", args.sr_iters, "--max-tokens", args.direct_max_tokens,
                            "--feedback-max-tokens", args.feedback_max_tokens,
                            *recover, *(["--recover-feedback"] if args.recover else []), *request_args(args)), args)
    elif method == "mad":
        repeat("MAD", lambda: missing(f["mad"], ids, args.k),
               lambda: call("mad.py", *common, "--out", f["mad"], "--k", args.k, "--agents", args.mad_agents,
                            "--rounds", args.mad_rounds, *recover, *request_args(args)), args)


def check_server(args) -> None:
    """The port serves the model with a window of at least WINDOW tokens."""
    name = MODELS[args.model]["name"]
    try:
        served = json.load(urllib.request.urlopen(endpoint(args) + "/v1/models", timeout=10))
    except OSError as err:
        raise SystemExit(f"no server on port {args.port} ({err})")
    found = {m.get("id"): m.get("max_model_len") for m in served.get("data", [])}
    if name not in found:
        raise SystemExit(f"port {args.port} serves {list(found)}, not {name}")
    if found[name] is not None and found[name] < WINDOW:
        raise SystemExit(f"{name} on port {args.port} has a {found[name]}-token window; the token limits "
                         f"assume {WINDOW} (the deploy scripts' --max-model-len)")


def check_judge_access() -> None:
    """HLE is graded by the judge model through the OpenAI API (judge_answers.py), after its replies
    are made. Check, without a paid call, that the key is in .env and the API's host can be reached.
    Only a warning: the replies are what costs time, and --score-only grades them later from a
    machine that has access."""
    import socket
    from urllib.parse import urlparse
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    problem = None
    if not os.environ.get("OPENAI_API_KEY"):
        problem = "OPENAI_API_KEY is not set (expected in .env)"
    else:
        host = urlparse(os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com").hostname
        try:
            socket.create_connection((host, 443), timeout=10).close()
        except OSError as err:
            problem = f"cannot reach {host}:443 ({err})"
    if problem:
        print(f"WARNING: HLE is graded by the judge model, and {problem}. Its replies will still be made; "
              "grading HLE will fail at the end of the HLE step (the other datasets still run). Grade it "
              "later with --score-only from a machine that has access.", flush=True)
    else:
        print("judge access for HLE: key found, API host reachable", flush=True)


# --- scoring -------------------------------------------------------------------------------------

def run_answer(rec: dict, item: dict) -> str | None:
    """A run's answer: its one reply's, or the vote over its replies (self-consistency, MAD)."""
    texts = rec["finals"] if "finals" in rec else [rec.get("content")]
    return tasks.vote([tasks.answer_of(t, item) for t in texts], item)


def grade_all(pairs: set, items: dict, args) -> dict:
    """(question id, answer) -> right or wrong; HLE through the judge model in one batch."""
    hle = sorted(p for p in pairs if tasks.kind(items[p[0]]) == "hle")
    verdict = {p: tasks.correct(items[p[0]], p[1]) for p in pairs if tasks.kind(items[p[0]]) != "hle"}
    if hle:
        from judge_answers import JUDGE_MODEL, Judge    # scripts/ is on the path (hle_format)
        judge = Judge(args.judge_cache, model=args.judge_model or JUDGE_MODEL)
        verdict.update(zip(hle, judge.grade_many([(items[q], a) for q, a in hle], workers=args.judge_workers)))
        print(f"judge {judge.model}: {len(hle)} distinct answers; {judge.stats}")
    return verdict


def score(args, methods: list[str], f: dict, items: list[dict]) -> tuple[str, dict]:
    by_id = {it["id"]: it for it in items}
    ids, k = set(by_id), args.k
    runs = {m: load_runs(f[m], ids, k) for m in methods}
    done = {m: {q for q in ids if all((q, r) in runs[m] for r in range(k))} for m in methods}
    common = sorted(set.intersection(*done.values())) if methods else []
    answers = {m: {q: [run_answer(runs[m][(q, r)], by_id[q]) for r in range(k)] for q in common} for m in methods}
    voted = {m: {q: tasks.vote(answers[m][q], by_id[q]) for q in common} for m in methods}
    pairs = {(q, a) for m in methods for q in common for a in answers[m][q] + [voted[m][q]] if a is not None}
    verdict = grade_all(pairs, by_id, args)

    def right(q, a):
        return a is not None and verdict[(q, a)]

    n = len(common)
    rows, detail = [], {}
    for m in methods:
        marks = {q: [right(q, a) for a in answers[m][q]] for q in common}
        tokens = [runs[m][(q, r)].get("completion_tokens") or 0 for q in common for r in range(k)]
        res = {"questions": n, "finished": len(done[m]),
               "avg@1": sum(marks[q][0] for q in common) / max(n, 1),
               f"vote@{k}": sum(right(q, voted[m][q]) for q in common) / max(n, 1),
               f"avg@{k}": sum(sum(marks[q]) / k for q in common) / max(n, 1),
               f"pass@{k}": sum(any(marks[q]) for q in common) / max(n, 1),
               "tokens": sum(tokens) / max(len(tokens), 1),
               "tokens_per_question": sum(tokens) / max(n, 1),         # all k runs of a question together
               "no_answer": sum(a is None for q in common for a in answers[m][q]) / max(n * k, 1)}
        rows.append((m, res))
        detail[m] = {"file": str(f[m]), "metrics": res,
                     "per_question": {q: {"answers": answers[m][q], "marks": [int(x) for x in marks[q]],
                                          "vote": voted[m][q], "vote_mark": int(right(q, voted[m][q]))}
                                      for q in common}}

    title = f"{Path(args.data).stem}{f' (first {args.limit})' if args.limit else ''}"
    lines = [f"## {title}: {MODELS[args.model]['name']}",
             "",
             f"{n} questions (every method finished them; {len(items)} in the file), {k} runs per question, "
             f"recovery {'on' if args.recover else 'off'}. Accuracy in %.",
             "",
             f"| Method | avg@1 | vote@{k} | avg@{k} | pass@{k} | tokens / run | tokens / question ({k} run{'s' if k != 1 else ''}) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m, res in rows:
        lines.append(f"| {label(m, args)} | {100 * res['avg@1']:.1f} | {100 * res[f'vote@{k}']:.1f} | "
                     f"{100 * res[f'avg@{k}']:.1f} | {100 * res[f'pass@{k}']:.1f} | {res['tokens'] / 1000:.1f}k | "
                     f"{res['tokens_per_question'] / 1000:.1f}k |")
    lines += ["",
              f"avg@1: run 1 alone. vote@{k}: the most common answer of the {k} runs (a tie goes to the earliest "
              f"run). avg@{k}: the mean of the {k} runs. pass@{k}: right in at least one run. tokens / run: "
              f"completion tokens of every call in one run, recovery included. tokens / question: the same "
              f"summed over the {k} runs of a question, then averaged over the questions."]
    notes = [f"{label(m, args)}: {len(items) - len(done[m])} questions not finished" for m in methods
             if len(done[m]) < len(items)]
    notes += [f"{label(m, args)}: no answer in {100 * res['no_answer']:.1f}% of runs" for m, res in rows
              if res["no_answer"] > 0]
    if notes:
        lines += [""] + [f"- {x}" for x in notes]
    return "\n".join(lines) + "\n", detail


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS), help="the model (see above)")
    p.add_argument("--data", required=True,
                   help="a dataset file, or all: the test splits of SuperGPQA, HLE, GPQA-Diamond and MATH level 5")
    p.add_argument("--port", type=int, default=None, help="the server's port on localhost (default: the model's)")
    p.add_argument("--methods", default=",".join(METHODS), help=f"comma-separated, from {','.join(METHODS)}")
    p.add_argument("--k", type=int, default=3, help="runs per question for every method")
    p.add_argument("--sc-n", type=int, default=5, help="self-consistency: replies voted over in one run")
    p.add_argument("--sr-iters", type=int, default=2,
                   help="Self-Refine: most feedback -> refine rounds (2, as the pipeline's; the paper: 4)")
    p.add_argument("--mad-agents", type=int, default=3)
    p.add_argument("--mad-rounds", type=int, default=2, help="MAD: rounds, the first answer included (paper: 2)")
    p.add_argument("--no-recover", dest="recover", action="store_false",
                   help="the methods as published: a reply cut off before its answer stays without one")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (a short check)")
    p.add_argument("--concurrency", type=int, default=64, help="requests in flight")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--score-only", action="store_true", help="no model calls: score what the files hold")
    p.add_argument("--results-dir", default=str(HERE / "results"))
    p.add_argument("--direct-max-tokens", type=int, default=28672)
    p.add_argument("--feedback-max-tokens", type=int, default=24576,
                   help="Self-Refine's feedback turn: the same room as its answer turns")
    p.add_argument("--python", default=VENV_PYTHON, help="the project's venv, which runs every step")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--judge-model", default=None, help="HLE: the judge (default judge_answers.JUDGE_MODEL)")
    p.add_argument("--judge-cache", default=None, help="HLE: the verdict cache (default: the search's)")
    p.add_argument("--judge-workers", type=int, default=16)
    args = p.parse_args()
    use_python(args.python)
    global tasks
    import tasks
    args.tag = MODELS[args.model]["tag"]
    if args.port is None:
        args.port = MODELS[args.model]["port"]

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if bad := [m for m in methods if m not in METHODS]:
        raise SystemExit(f"unknown methods {bad}; choose from {METHODS}")
    methods = [m for m in METHODS if m in methods]               # the table's order
    datasets = [str(ROOT / d) for d in TEST_SETS] if args.data == "all" else [args.data]
    for data in datasets:                                        # a missing file or unknown dataset stops here
        for it in json.loads(Path(data).read_text()):
            tasks.kind(it)
    os.makedirs(args.results_dir, exist_ok=True)
    if not args.score_only:
        check_server(args)
    if any("hle" in Path(d).stem for d in datasets):
        check_judge_access()

    tables, failed, stopped = [], [], None
    for i, data in enumerate(datasets):
        one = argparse.Namespace(**{**vars(args), "data": data})
        if len(datasets) > 1:
            print(f"\n########## {Path(data).stem}", flush=True)
        try:
            tables.append(run_dataset(one, methods))
        except ServerDown as err:                                # the later datasets would fail too: stop
            if len(datasets) == 1:
                raise SystemExit(str(err)) from None
            stopped = str(err)
            failed.append(f"{Path(data).stem}: {err}")
            failed += [f"{Path(d).stem}: not run (the server stopped answering)" for d in datasets[i + 1:]]
            print(f"{Path(data).stem} stopped: {err}", flush=True)
            break
        except (Exception, SystemExit) as err:                   # not KeyboardInterrupt
            if len(datasets) == 1:
                raise
            failed.append(f"{Path(data).stem}: {type(err).__name__}: {err}")   # the other datasets still run
            print(f"{Path(data).stem} stopped: {type(err).__name__}: {err}", flush=True)
    if len(datasets) > 1:
        text = "\n".join(tables) + "".join(f"\n- not finished: {x}\n" for x in failed)
        stem = str(Path(args.results_dir) / f"table_{args.tag}_all") + suffix(args)
        Path(stem + ".md").write_text(text)
        print("\n" + text + f"\nall tables written to {stem}.md", flush=True)
    if stopped:
        raise SystemExit(stopped)
    if failed:
        raise SystemExit(f"{len(failed)} dataset(s) not finished; run the same command again")


def suffix(args) -> str:
    return ("_rec" if args.recover else "") + (f"_first{args.limit}" if args.limit else "")


def run_dataset(args, methods: list[str]) -> str:
    """Run the methods on args.data, one after another, then score them; returns the table."""
    items = json.loads(Path(args.data).read_text())
    if args.limit:
        items = items[: args.limit]
    f = files(args)
    ids = {it["id"] for it in items}
    if not args.score_only:
        for m in methods:
            print(f"\n##### {label(m, args)}", flush=True)
            run_method(m, args, f, ids)
    table, detail = score(args, methods, f, items)
    stem = str(Path(args.results_dir) / f"table_{args.tag}_{Path(args.data).stem}") + suffix(args)
    Path(stem + ".md").write_text(table)
    Path(stem + ".json").write_text(json.dumps({"args": vars(args), "methods": detail}, indent=1,
                                               ensure_ascii=False))
    print("\n" + table)
    print(f"written to {stem}.md (+ .json)", flush=True)
    return table


if __name__ == "__main__":
    main()

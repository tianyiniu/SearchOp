"""Offline checks for the external baselines on all four kinds of dataset: the answer rules
(tasks.py), and a whole run_baselines.py run (direct, self-consistency, Self-Refine, MAD, with
recovery) against a fake OpenAI-compatible server that answers by a fixed rule. Checks the
requests each method sends, the debate's flow, the recovery calls, resuming without new calls,
and the table's numbers against an independent count. No GPU, no network.

    /nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python baselines/test_baselines.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path

from aiohttp import web

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import tasks  # noqa: E402

TAG = "qwen35_9b_think"           # run_baselines.MODELS["qwen35-9b"]["tag"]


# --- the answer rules ----------------------------------------------------------------------------

def test_rules():
    gpqa = {"id": "g", "question": "Q?", "options": ["w", "x", "y", "z"], "answer_letter": "C", "dataset": "gpqa"}
    math = {"id": "m", "question": "P?", "options": [], "answer": "\\frac{1}{2}", "dataset": "math"}
    sgpqa = {"id": "s", "question": "Q?", "options": list("abcdefghij"), "answer_letter": "E"}
    hle = {"id": "h", "question": "Q?", "options": [], "answer": "7", "answer_type": "exactMatch"}
    assert [tasks.kind(r) for r in (gpqa, math, sgpqa, hle)] == ["gpqa", "math", "supergpqa", "hle"]
    try:
        tasks.kind({"id": "x", "dataset": "nope"})
        raise AssertionError("an unknown dataset must stop")
    except ValueError:
        pass
    # letters: GPQA takes only A-D
    assert tasks.answer_of("so\nAnswer: C", gpqa) == "C"
    assert tasks.answer_of("so\nAnswer: E", gpqa) is None
    assert tasks.answer_of("so\nAnswer: E", sgpqa) == "E"
    assert tasks.correct(gpqa, "C") and not tasks.correct(gpqa, "B") and not tasks.correct(gpqa, None)
    # math: the last \boxed{}, nested braces, \fbox, the no-brace form, a box cut off
    assert tasks.last_boxed("x \\boxed{1} then \\boxed{\\frac{3}{4}}.") == "\\frac{3}{4}"
    assert tasks.last_boxed("\\fbox{5}") == "5"
    assert tasks.last_boxed("so $\\boxed 12$ done") == "12"
    assert tasks.last_boxed("\\boxed{\\frac{1}{2") is None and tasks.last_boxed("none") is None
    assert tasks.last_boxed("\\boxed{}") is None
    assert tasks.correct(math, "0.5") and tasks.correct(math, "\\dfrac12") and not tasks.correct(math, "2")
    # votes: None does not vote, a tie goes to the first given, math-verify merges equal answers
    assert tasks.vote(["B", None, "C", "C", "B"], gpqa) == "B"
    assert tasks.vote([None, None], gpqa) is None
    assert tasks.vote(["3", "0.5", "\\frac{1}{2}"], math) == "0.5"
    assert tasks.vote(["3", "0.5"], math) == "3"
    # later turns: SuperGPQA's prompts are the ones the scripts always used; GPQA and math have theirs
    assert tasks.refine_prompt(sgpqa).endswith("one of A, B, C, D, E, F, G, H, I, or J.")
    assert tasks.refine_prompt(gpqa).endswith("where LETTER is one of ABCD.")
    assert tasks.refine_prompt(math).endswith("\\boxed{}.") and "options" not in tasks.feedback_prompt(math)
    assert tasks.recover_prompts(gpqa)[0].endswith("one of ABCD.") and "\\boxed{}" in tasks.recover_prompts(math)[0]
    msg = tasks.debate_message(gpqa, ["one", "two"])
    assert msg.startswith(tasks.DEBATE_PREFIX) and msg.count("One agent solution: ```") == 2
    assert "Examine your solution and that other agents step by step." in msg and msg.endswith("one of ABCD.")
    msg = tasks.debate_message(math, ["r1 {x}", "r2"])
    assert "The original math problem is P?." in msg and "```r1 {x}```" in msg and "\\boxed{answer}" in msg
    assert tasks.debate_message(gpqa, []).startswith("Can you double check that your answer is correct.")
    # Self-Refine's stop rule reads through LaTeX: the math prompt makes the model box its verdict
    assert tasks.says_correct("ok.\n\\boxed{it \\ is \\ correct}") and tasks.says_correct("**It is correct.**")
    assert not tasks.says_correct("it is not correct") and not tasks.says_correct(None)
    # first requests
    assert "A) w\nB) x\nC) y\nD) z" in tasks.messages(gpqa)[0]["content"]
    assert tasks.messages(math)[0]["content"] == "P?\n" + tasks.MATH_INSTRUCTION
    assert tasks.messages(hle)[0]["role"] == "system"
    print("rules: ok")


# --- the fake server ---------------------------------------------------------------------------------

def gold_math(question: str) -> str:
    return question.split("=")[-1].strip()


def boxed(a):
    return f"so \\boxed{{{a}}}"


class Fake:
    """Replies by a fixed rule on the request's seed and its last user turn:
    first request: letter A/B/C (or a math value) by seed; one in seven is cut off with no answer;
    feedback: 'it is correct' for even seeds; refine: a letter by seed; debate turn: the most common
    answer among the other agents' replies shown; recovery: an answer."""

    def __init__(self):
        self.requests = []
        self.rejected = 0
        self.lock = threading.Lock()

    def answer(self, kind, seed, question):
        if kind == "math":
            g = gold_math(question)
            return boxed([g, "0.5" if g == "\\frac{1}{2}" else g, "99"][seed % 3])
        return f"Reasoning.\nAnswer: {'ABC'[seed % 3]}"

    def reply(self, payload):
        msgs = payload["messages"]
        last = msgs[-1]["content"]
        first = next(m["content"] for m in msgs if m["role"] == "user")
        kind = "math" if tasks.MATH_INSTRUCTION in first else "mc"
        question = first.split("\n")[0]
        seed = payload["seed"]
        if last == tasks.FEEDBACK_RECOVER:                   # a cut-off feedback, recovered
            return "it is correct", "stop"
        if last.startswith("Your response above ran out") or last.startswith("STOP."):
            return self.answer(kind, 1, question), "stop"
        if last.startswith("There may be an error"):
            if seed % 5 == 0:                                # thinks until the limit, no text
                return "", "length"
            ok = "\\boxed{it \\ is \\ correct}" if kind == "math" else "it is correct"     # as Qwen writes it
            return (ok if seed % 2 == 0 else "Step 2 is wrong."), "stop"
        if last.startswith(tasks.DEBATE_PREFIX):
            shown = re.findall(r"```(.*?)```", last, re.S)
            ans = [tasks.last_boxed(s) if kind == "math" else (re.findall(r"Answer: ([A-J])", s) or [None])[-1]
                   for s in shown]
            ans = [a for a in ans if a]
            if ans:
                a = Counter(ans).most_common(1)[0][0]
                return (boxed(a) if kind == "math" else f"Agreed.\nAnswer: {a}"), "stop"
        if seed % 7 == 0 and not last.startswith("Using the feedback"):
            return "", "length"
        return self.answer(kind, seed, question), "stop"

    @staticmethod
    def count(messages):                         # the fake's token count: 4 characters a token
        return len(json.dumps(messages)) // 4

    async def tokenize(self, request):
        body = await request.json()
        return web.json_response({"count": self.count(body["messages"]), "max_model_len": 32768})

    async def chat(self, request):
        payload = await request.json()
        n = self.count(payload["messages"])
        if n + payload["max_tokens"] > 32768:    # vLLM's rejection: only a lower bound on the prompt
            with self.lock:
                self.rejected += 1
            return web.json_response({"error": {"message": "This model's maximum context length is 32768 tokens. "
                                                f"However, you requested {payload['max_tokens']} output tokens and "
                                                f"your prompt contains at least {min(n, 4097)} input tokens"}},
                                     status=400)
        content, finish = self.reply(payload)
        with self.lock:
            self.requests.append(payload)
        msg = {"role": "assistant", "content": content, "reasoning_content": "thinking " * 5}
        return web.json_response({"choices": [{"message": msg, "finish_reason": finish}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 100}})

    async def models(self, request):
        from run_baselines import MODELS
        return web.json_response({"data": [{"id": m["name"], "max_model_len": 32768} for m in MODELS.values()]})


def serve(fake):
    import asyncio
    loop = asyncio.new_event_loop()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", fake.chat)
    app.router.add_get("/v1/models", fake.models)
    app.router.add_post("/tokenize", fake.tokenize)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "127.0.0.1", 0)
    loop.run_until_complete(site.start())
    port = site._server.sockets[0].getsockname()[1]
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return port


# --- a whole run -------------------------------------------------------------------------------------

def datasets(tmp: Path) -> dict:
    gpqa = json.loads((ROOT / "datasets/gpqa_diamond_test.json").read_text())[:4]
    sgpqa = json.loads((ROOT / "datasets/supergpqa_600_test.json").read_text())[:3]
    math = [{"id": f"m{i}", "question": f"Compute x = {g}", "options": [], "answer": g, "dataset": "math",
             "discipline": "Algebra", "field": "Algebra", "subfield": "Algebra", "difficulty": "Level 5"}
            for i, g in enumerate(["\\frac{1}{2}", "4", "\\sqrt{2}", "\\frac{1}{2}"])]
    out = {}
    for name, rows in (("gpqa", gpqa), ("math", math), ("supergpqa", sgpqa)):
        out[name] = tmp / f"tiny_{name}.json"
        out[name].write_text(json.dumps(rows))
    return out


def run(port, data, results, *extra, model="qwen35-9b", python=sys.executable):
    cmd = [python, str(HERE / "run_baselines.py"), "--model", model, "--port", str(port), "--data", str(data),
           "--results-dir", str(results), "--concurrency", "8", *extra]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:])
        raise AssertionError(f"run_baselines.py failed on {data.name}")
    return res.stdout


def expected(results: Path, data: Path, k=3, limit=0, hle_right="B") -> dict:
    """The table's numbers by an independent count over the result files. HLE: the filled judge
    cache calls `hle_right` right and every other answer wrong."""
    rows = json.loads(data.read_text())
    items = {r["id"]: r for r in (rows[:limit] if limit else rows)}
    is_math = any(r.get("dataset") == "math" for r in items.values())
    is_hle = any("answer_type" in r for r in items.values())
    canon = {"\\frac{1}{2}": "1/2", "0.5": "1/2", "4": "4", "\\sqrt{2}": "r2", "99": "99"}

    def read(text, it):
        if is_math:
            m = re.findall(r"\\boxed\{(.*?)\}$", (text or "").strip())
            return canon[m[-1]] if m else None
        m = re.findall(r"Answer: ([A-D])\s*$", (text or "").strip())
        return m[-1] if m else None

    def maj(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        c = Counter(xs)
        best = max(c.values())
        return next(x for x in xs if c[x] == best)

    def gold(it):
        return hle_right if is_hle else canon[it["answer"]] if is_math else it["answer_letter"]

    out = {}
    stem = f"{TAG}_{data.stem}_rec"
    for m, fname in (("direct", f"direct_{stem}"), ("sc", f"sc5_{stem}"), ("selfrefine", f"selfrefine_it2_{stem}"),
                     ("mad", f"mad_a3r2_{stem}")):
        recs = {}
        for line in open(results / f"{fname}.jsonl"):
            r = json.loads(line)
            if r.get("error") is None and r["sample_idx"] < k:
                recs[(r["id"], r["sample_idx"])] = r
        a1 = v = a3 = p3 = 0
        for q, it in items.items():
            runs = [maj([read(t, it) for t in (recs[(q, r)].get("finals") or [recs[(q, r)]["content"]])])
                    for r in range(k)]
            marks = [x == gold(it) for x in runs]
            a1 += marks[0]
            v += maj(runs) == gold(it)
            a3 += sum(marks) / k
            p3 += any(marks)
        n = len(items)
        out[m] = (round(100 * a1 / n, 1), round(100 * v / n, 1), round(100 * a3 / n, 1), round(100 * p3 / n, 1))
    return out


def token_runs(results: Path, method: str, data: Path, k=3) -> list[list[int]]:
    """Each question's completion tokens in its k runs, read from the method's file."""
    fname = {"direct": "direct", "sc": "sc5", "selfrefine": "selfrefine_it2", "mad": "mad_a3r2"}[method]
    recs = {}
    for line in open(results / f"{fname}_{TAG}_{data.stem}_rec.jsonl"):
        r = json.loads(line)
        if r.get("error") is None and r["sample_idx"] < k:
            recs[(r["id"], r["sample_idx"])] = r["completion_tokens"]
    return [[recs[(r["id"], i)] for i in range(k)] for r in json.loads(data.read_text())]


def table_numbers(md: str) -> dict:
    rows = {}
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 7 and cells[1].replace(".", "").isdigit():
            key = {"Direct CoT": "direct", "Self-consistency": "sc", "Self-Refine": "selfrefine",
                   "MAD": "mad"}[cells[0].split(" (")[0]]
            rows[key] = tuple(float(c) for c in cells[1:5])
    return rows


def test_window(port, fake):
    """A request too long for the window is sent again with max_tokens cut to the room left."""
    import asyncio

    import aiohttp

    from generate import CONTEXT_MARGIN, request_one

    async def go():
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        out = []
        async with aiohttp.ClientSession() as s:
            for chars in (60000, 400, 140000):                  # too long, fits, no room left
                payload = {"model": "Qwen/Qwen3.5-9B", "max_tokens": 28672, "seed": 3,
                           "messages": [{"role": "user", "content": "x" * chars + "\nCompute x = 4\n"
                                         + tasks.MATH_INSTRUCTION}]}
                out.append((Fake.count(payload["messages"]), await request_one(s, url, payload, 3)))
        return out
    (n1, (d1, e1)), (_, (d2, e2)), (_, (d3, e3)) = asyncio.run(go())
    assert e1 is None and d1["max_tokens_cut"] == 32768 - n1 - CONTEXT_MARGIN, (e1, d1)
    assert e2 is None and "max_tokens_cut" not in d2
    assert e3 is not None and e3.startswith("no room") and "maximum context length" in e3, e3
    assert fake.rejected >= 2
    print("window: a too-long request is cut to fit; one that fits is unchanged; no room returns 'no room'")


def test_sr_no_room(port, fake):
    """Self-Refine: a FEEDBACK or REFINE turn with no room left in the window ends the loop with the
    answer so far (no error); the question lengths are chosen so that each case happens."""
    from generate import MIN_ROOM, CONTEXT_MARGIN, question_messages, sample_seed
    full = 32768 - MIN_ROOM - CONTEXT_MARGIN             # the longest prompt that still gets a reply

    def item(qid, pad):
        return {"id": qid, "question": "Compute x = 4\n" + "x" * pad, "options": [], "answer": "4",
                "dataset": "math", "discipline": "Algebra", "field": "Algebra", "subfield": "Algebra",
                "difficulty": "Level 5"}

    def ids(ok):                                          # ids whose seeds give the fake's replies we need
        return next(f"nr{i}" for i in range(10000) if ok(sample_seed(f"nr{i}", 0)))

    def pad_for(qid, turn_msgs, target):                 # the padding that makes those messages count `target`
        lo, hi = 0, 4 * 33000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if Fake.count(turn_msgs(item(qid, mid))) <= target: lo = mid
            else: hi = mid - 1
        return lo

    def init_reply(it):
        return fake.answer("math", sample_seed(it["id"], 0), "Compute x = 4")

    def fb_msgs(it):
        return question_messages(it) + [{"role": "assistant", "content": init_reply(it)},
                                        {"role": "user", "content": tasks.feedback_prompt(it)}]

    def rf_msgs(it):
        return fb_msgs(it) + [{"role": "assistant", "content": "Step 2 is wrong."},
                              {"role": "user", "content": tasks.refine_prompt(it)}]

    # 1: the first answer fits, the FEEDBACK turn does not
    q1 = ids(lambda s: s % 7)
    a = item(q1, pad_for(q1, question_messages, full))
    assert Fake.count(fb_msgs(a)) > full
    # 2: the FEEDBACK turn fits and finds an error; the REFINE turn does not
    q2 = ids(lambda s: s % 7 and (s + 1000) % 5 and (s + 1000) % 2)
    b = item(q2, pad_for(q2, fb_msgs, full))
    assert Fake.count(fb_msgs(b)) <= full < Fake.count(rf_msgs(b))
    with tempfile.TemporaryDirectory() as tmp:
        data, out = Path(tmp) / "long.json", Path(tmp) / "sr.jsonl"
        data.write_text(json.dumps([a, b]))
        res = subprocess.run([sys.executable, str(HERE / "selfrefine.py"), "--data", str(data), "--out", str(out),
                              "--endpoints", f"http://127.0.0.1:{port}", "--model", "Qwen/Qwen3.5-9B", "--k", "1",
                              "--family", "qwen", "--max-iters", "2", "--max-tokens", "28672",
                              "--feedback-max-tokens", "24576", "--recover", "--recover-feedback"],
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stderr[-2000:]
        recs = {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}
    r1, r2 = recs[q1], recs[q2]
    for r, ended, roles in ((r1, "feedback", ["init"]), (r2, "refine", ["init", "feedback"])):
        sr = r["selfrefine"]
        assert r["error"] is None, r["error"]
        assert sr["no_room"] == ended and not sr["stopped_early"] and sr["n_refinements"] == 0, sr
        assert [t["role"] for t in sr["turns"]] == roles, sr["turns"]
        assert r["content"] == sr["init_answer"] == init_reply(a if r is r1 else b), r["content"]
    assert sr["turns"][1]["content"] == "Step 2 is wrong."
    assert "ended for want of room 2, errors 0" in res.stdout, res.stdout[-500:]
    print("Self-Refine: no room for FEEDBACK or REFINE ends the loop with the answer so far")


def test_no_room(port, fake):
    """Every method gives each question a result when a call has no room in the window: Direct CoT
    a sample with no answer, recovery the reply as it was cut off, Self-Refine's first answer a run
    with no answer, a MAD agent its reply of the round before (none in round 1)."""
    from generate import MIN_ROOM, CONTEXT_MARGIN, question_messages, sample_seed
    full = 32768 - MIN_ROOM - CONTEXT_MARGIN             # the longest prompt that still gets a reply

    def item(qid, pad):
        return {"id": qid, "question": "Compute x = 4\n" + "x" * pad, "options": [], "answer": "4",
                "dataset": "math", "discipline": "Algebra", "field": "Algebra", "subfield": "Algebra",
                "difficulty": "Level 5"}

    def pad_for(qid, target):                            # the padding that makes the question count `target`
        lo, hi = 0, 4 * 33000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if Fake.count(question_messages(item(qid, mid))) <= target: lo = mid
            else: hi = mid - 1
        return lo

    cut_id = next(f"cut{i}" for i in range(10000) if sample_seed(f"cut{i}", 0) % 7 == 0)   # the fake cuts it off
    too_long = item("long", pad_for("long", full + 50))                # no room for any reply
    no_recovery = item(cut_id, pad_for(cut_id, full - 1))              # a reply fits; its recovery does not
    debate = item("deb", pad_for("deb", full - 50))                    # round 1 fits; round 2 does not
    py = [sys.executable]
    common = ["--endpoints", f"http://127.0.0.1:{port}", "--model", "Qwen/Qwen3.5-9B", "--family", "qwen", "--k", "1"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def call(script, rows, *extra):
            data, out = tmp / f"{script}.json", tmp / f"{script}.jsonl"
            data.write_text(json.dumps(rows))
            res = subprocess.run(py + [str(HERE / script), "--data", str(data), "--out", str(out), *common, *extra],
                                 capture_output=True, text=True)
            assert res.returncode == 0, (script, res.stderr[-2000:])
            return {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}, res.stdout

        direct, out = call("generate.py", [too_long, no_recovery], "--max-tokens", "28672")
        assert "no room=1, errors=0" in out, out
        d = direct["long"]
        assert d["error"] is None and d["content"] is None and d["finish_reason"] == "no_room", d
        assert direct[cut_id]["finish_reason"] == "length" and direct[cut_id]["max_tokens_cut"] == 1025
        data = tmp / "direct.json"
        data.write_text(json.dumps([too_long, no_recovery]))
        res = subprocess.run(py + [str(HERE / "recover.py"), "--data", str(data), "--results", str(tmp / "generate.py.jsonl"),
                                   "--out", str(tmp / "rec.jsonl"), *[a for a in common if a not in ("--k", "1")]],
                             capture_output=True, text=True)
        assert res.returncode == 0 and "1 had no room to ask, 0 failed" in res.stdout, (res.stdout, res.stderr[-1500:])
        rec = {r["id"]: r for r in map(json.loads, (tmp / "rec.jsonl").read_text().splitlines())}
        assert set(rec) == {"long", cut_id}, rec.keys()                   # both questions have a result
        r = rec[cut_id]
        assert r["error"] is None and r["recovery"]["no_room"] and r["recovery"]["calls"] == 0, r["recovery"]
        assert tasks.answer_of(r["content"], no_recovery) is None

        sr, out = call("selfrefine.py", [too_long], "--max-iters", "2", "--recover", "--recover-feedback")
        s = sr["long"]
        assert s["error"] is None and s["content"] is None and s["selfrefine"]["no_room"] == "init", s
        assert "ended for want of room 1, errors 0" in out, out

        mad, out = call("mad.py", [too_long, debate], "--agents", "3", "--rounds", "2")
        assert "turns with no room 9, errors 0" in out, out      # "long": 3 + 3, "deb": 3 in round 2
        m = mad["long"]
        assert m["error"] is None and m["finals"] == [None] * 3, m
        m = mad["deb"]
        first, second = m["mad"]["turns"]
        assert m["error"] is None and m["finals"] == [t["content"] for t in first], m
        assert [t["finish_reason"] for t in second] == ["no_room"] * 3 and sum(t["completion_tokens"] for t in second) == 0
    print("no room: Direct CoT, recovery, Self-Refine and MAD give every question a result")


def test_run():
    fake = Fake()
    port = serve(fake)
    test_window(port, fake)
    test_sr_no_room(port, fake)
    test_no_room(port, fake)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = datasets(tmp)
        results = tmp / "results"
        recovered_feedback = 0
        for name, path in data.items():
            before = len(fake.requests)
            out = run(port, path, results)
            reqs = fake.requests[before:]
            md = (results / f"table_{TAG}_{path.stem}_rec.md").read_text()
            # the table: four rows in order, numbers equal to the independent count
            order = [line.split("|")[1].strip().split(" (")[0] for line in md.splitlines() if line.startswith("| ")][1:]
            assert order == ["Direct CoT", "Self-consistency", "Self-Refine", "MAD"], order
            assert table_numbers(md) == expected(results, path), (name, table_numbers(md), expected(results, path))
            assert "4 questions" in md or "3 questions" in md, md
            saved = json.loads((results / f"table_{TAG}_{path.stem}_rec.json").read_text())["methods"]
            for m, d in saved.items():            # tokens per question: the 3 runs' tokens together
                per_q = [sum(runs) for runs in token_runs(results, m, path)]
                assert abs(d["metrics"]["tokens_per_question"] - sum(per_q) / len(per_q)) < 1e-9, m
                assert abs(d["metrics"]["tokens_per_question"] - 3 * d["metrics"]["tokens"]) < 1e-9, m
            assert "tokens / question (3 runs)" in md
            # the requests: every first request is the dataset's prompt
            rows = json.loads(path.read_text())
            firsts = {tasks.messages(r)[0]["content"] for r in rows}
            assert all(q["messages"][0]["content"] in firsts for q in reqs), name
            assert not any(tasks.EXPLAIN_SENTENCE in q["messages"][0]["content"] for q in reqs)   # Qwen: unchanged
            # direct + self-consistency share samples: 15 per question, seeds 0..14 of each question
            n_q = len(rows)
            raw = [json.loads(x) for x in open(results / f"direct_{TAG}_{path.stem}.jsonl")]
            assert sorted((r["id"], r["sample_idx"]) for r in raw) == sorted(
                (r["id"], i) for r in rows for i in range(15)), name
            # MAD: 3 agents x 2 rounds x 3 runs per question; round 2 shows the 2 other agents
            debate = [q for q in reqs if q["messages"][-1]["content"].startswith(tasks.DEBATE_PREFIX)]
            assert len(debate) == n_q * 3 * 3, (name, len(debate))
            for q in debate:
                assert q["messages"][-1]["content"].count("One agent solution: ```") == 2
                assert [m["role"] for m in q["messages"]] == ["user", "assistant", "user"]
                assert q["max_tokens"] == 24576
            # recovery happened (one in seven first replies is cut off) and went out with thinking off
            rec = [q for q in reqs if q["messages"][-1]["content"].startswith("Your response above ran out")]
            assert rec and all(q["chat_template_kwargs"] == {"enable_thinking": False} for q in rec)
            # Self-Refine: up to 2 rounds (run_baselines.py's default); feedback and refine prompts are the dataset's
            fb = [q for q in reqs if q["messages"][-1]["content"].startswith("There may be an error")]
            assert fb and all(q["messages"][-1]["content"] == tasks.feedback_prompt(rows[0]) for q in fb)
            assert max(len(q["messages"]) for q in fb) <= 2 + 4 * 2
            assert all(q["max_tokens"] == 24576 for q in fb)     # feedback gets the answer turns' room
            for line in open(results / f"selfrefine_it2_{TAG}_{path.stem}_rec.jsonl"):
                turns = json.loads(line)["selfrefine"]["turns"]
                for t in turns:                                  # a cut-off feedback was recovered
                    if t["role"] == "feedback" and t["finish_reason"] == "length":
                        assert t["recovery"]["calls"] == 1 and t["content"] == "it is correct", t
                        recovered_feedback += 1
                # a feedback that approves ends the loop
                assert all(t["role"] != "feedback" or not tasks.says_correct(t["content"]) or i == len(turns) - 1
                           for i, t in enumerate(turns)), (name, [t["role"] for t in turns])
            # resuming makes no new request and gives the same table
            before = len(fake.requests)
            run(port, path, results)
            assert len(fake.requests) == before, f"{name}: a rerun sent {len(fake.requests) - before} requests"
            assert (results / f"table_{TAG}_{path.stem}_rec.md").read_text() == md
            print(f"{name}: {len(reqs)} requests; table matches the independent count\n{md}")
        assert recovered_feedback > 0, "no feedback turn was cut off: the recovery went untested"
        print(f"feedback recovery: {recovered_feedback} cut-off feedback turns recovered")
        # --score-only with some methods, started from the system python3: it restarts in the venv
        out = run(port, data["gpqa"], results, "--score-only", "--methods", "direct,mad", python="/usr/bin/python3")
        assert "| Direct CoT |" in out and "| MAD (" in out and "Self-Refine" not in out
        # gpt-oss: reasoning effort high on every request, low on recovery, no Qwen fields
        before = len(fake.requests)
        run(port, data["gpqa"], tmp / "results_gptoss", model="gptoss-20b")
        reqs = fake.requests[before:]
        rec = [q for q in reqs if q["messages"][-1]["content"].startswith("Your response above ran out")]
        assert rec and all(q["reasoning_effort"] == "low" for q in rec)
        assert all(q["model"] == "openai/gpt-oss-20b" and "chat_template_kwargs" not in q and "top_k" not in q
                   for q in reqs)
        assert all(q["reasoning_effort"] == "high" and q["top_p"] == 1.0 for q in reqs if q not in rec)
        assert (tmp / "results_gptoss" / "table_gptoss20b_high_explain_tiny_gpqa_rec.md").exists()
        # gpt-oss's first prompt asks for the reasoning in the reply, in every request (recovery included)
        assert all(q["messages"][0]["content"].endswith("\n\n" + tasks.EXPLAIN_SENTENCE) for q in reqs)
        print(f"gpt-oss settings: ok ({len(reqs)} requests)")
        # --data all: the four test splits, first 2 questions each; HLE is graded from a judge cache
        # filled here, and the API key is blanked, so a missing verdict fails instead of calling the API
        hle_rows = json.loads((ROOT / "datasets/hle_text_test_200.json").read_text())[:2]
        cache = tmp / "judge.jsonl"
        cache.write_text("".join(json.dumps({"qid": r["id"], "answer": a, "correct": a == "B", "model": "gpt-6-luna",
                                             "effort": "medium", "v": "hle1"}) + "\n"
                                 for r in hle_rows for a in "ABC"))
        env = {**os.environ, "OPENAI_API_KEY": "", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}
        res = subprocess.run([sys.executable, str(HERE / "run_baselines.py"), "--model", "qwen35-9b", "--port",
                              str(port), "--data", "all", "--limit", "2", "--results-dir", str(tmp / "results_all"),
                              "--concurrency", "8", "--judge-cache", str(cache)], env=env, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
        md = (tmp / "results_all" / f"table_{TAG}_all_rec_first2.md").read_text()
        assert re.findall(r"^## (\S+) \(first 2\)", md, re.M) == [
            "supergpqa_2k_test", "hle_text_test_200", "gpqa_diamond_test", "math_l5_test"], md
        hle_md = (tmp / "results_all" / f"table_{TAG}_hle_text_test_200_rec_first2.md").read_text()
        want = expected(tmp / "results_all", ROOT / "datasets/hle_text_test_200.json", limit=2)
        assert table_numbers(hle_md) == want, (table_numbers(hle_md), want)
        assert "'calls': 0" in res.stdout, "the judge must not be called"
        print(f"--data all: 4 tables in order; HLE matches the independent count {want}")
        # the probe passes on both kinds of server and saves every request and reply
        for model, extra in (("qwen35-9b", ["--e2e"]), ("gptoss-20b", [])):
            out = tmp / f"probe_{model}"
            res = subprocess.run(["/usr/bin/python3", str(HERE / "probe_server.py"), "--model", model,
                                  "--port", str(port), "--out", str(out), *extra], capture_output=True, text=True)
            assert res.returncode == 0 and "FAIL" not in res.stdout, res.stdout[-3000:] + res.stderr[-2000:]
            rec = json.loads((out / "probe.json").read_text())
            names = ("supergpqa", "gpqa", "math", "hle-mc", "hle-open")
            want = {f"{c}: {n}" for c in ("first", "recovery", "feedback", "debate") for n in names}
            assert {r["check"] for r in rec["requests"]} == want, sorted({r["check"] for r in rec["requests"]})
            assert all(r["reply"] and r["error"] is None for r in rec["requests"])
            assert len(rec["checks"]) == 1 + len(want) + (3 if extra else 0) and all(c["ok"] for c in rec["checks"])
            assert (out / "probe.log").read_text().count("PASS") == len(rec["checks"])
            if extra:
                assert len(list((out / "e2e").glob("table_*_first1.md"))) == 3
                assert len(list((out / "e2e").glob("mad_*.jsonl"))) == 3
            print(f"probe {model}: {len(rec['checks'])} checks passed, {len(rec['requests'])} requests saved")


def test_server_down():
    """A server that stops answering stops the whole run at once: the next attempt is not made and the
    datasets not reached are marked 'not run' (in the gpt-oss run of 2026-10-05 the server died after
    SuperGPQA's replies, and the runner then spent 3 attempts of refused connections on each of the
    other 3 datasets). Failed requests while the server answers still mean 'run the same command again'."""
    import socket
    import run_baselines as RB
    s = socket.socket()
    s.bind(("localhost", 0))
    dead = s.getsockname()[1]
    s.close()                                     # nothing listens on this port now
    live = serve(Fake())
    attempts = []
    args = lambda port: type("A", (), {"port": port, "attempts": 3})()
    try:
        RB.repeat("x", lambda: 5, lambda: attempts.append(1), args(dead))
        assert False, "no ServerDown"
    except RB.ServerDown as err:
        assert "stopped answering" in str(err) and not attempts, (err, attempts)
    try:
        RB.repeat("x", lambda: 5, lambda: attempts.append(1), args(live))
        assert False, "no SystemExit"
    except SystemExit as err:
        assert "still missing after 3 attempts" in str(err) and len(attempts) == 3, (err, attempts)

    with tempfile.TemporaryDirectory() as tmp:
        calls = []

        def run_dataset(one, methods):
            calls.append(Path(one.data).stem)
            if len(calls) == 1:
                return "| a finished table |"
            RB.repeat("direct samples (k=3)", lambda: 5, lambda: attempts.append(1), one)   # the server is gone
        saved = (RB.run_dataset, RB.check_server, RB.check_judge_access, sys.argv)
        RB.run_dataset, RB.check_server, RB.check_judge_access = run_dataset, lambda a: None, lambda: None
        sys.argv = ["run_baselines.py", "--model", "qwen35-9b", "--data", "all", "--port", str(dead),
                    "--results-dir", tmp]
        n = len(attempts)
        try:
            RB.main()
            assert False, "the run did not stop"
        except SystemExit as err:
            assert "stopped answering" in str(err), err
        finally:
            RB.run_dataset, RB.check_server, RB.check_judge_access, sys.argv = saved
        assert len(calls) == 2 and len(attempts) == n, (calls, attempts)
        table = next(Path(tmp).glob("table_*_all*.md")).read_text()
        assert table.count("not run (the server stopped answering)") == len(RB.TEST_SETS) - 2, table
    print("server down: the run stops at once and marks the datasets not reached")


if __name__ == "__main__":
    test_rules()
    test_server_down()
    test_run()
    print("all baseline checks passed")

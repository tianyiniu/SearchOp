"""Choosers on fixed answer pools: which way of picking among the answers of a debate
gets the right one, measured on recorded pools so every chooser sees the same answers.

A pool is the blind first round(s) of a recorded debate, read from the search's round
cache (train) or the test evaluation's (test); no speaker is re-run:

    3H      three high-effort solvers              (solver_x3|high)
    4H      four high-effort solvers               (solver_x4|high)
    2H2L    two high-effort and two low-effort     (solver_x2|high > solver_x2|blind)

Choosers run on the split pools (two or more distinct answers); a unanimous pool keeps
its answer. Each candidate answer is argued by its first speaker in the pool (high
effort first), shown as the summary later speakers read (--show summary) or as
everything recorded of it, reply and summary (--show full):

    plurality     the executor's vote: most common answer, ties to the first committed
    judge         the existing judge round (judge|high after the pool, as the judge seeds
                  run it), through the round runner: recorded ones are replayed for free
    judge_blind   the judge's prompt, one unlabelled response per distinct answer in a
                  shuffled order: no speaker names, no counts
    pairwise      a knockout: the plurality answer holds, challengers in vote order; a match
                  is two calls (both orders) of a two-response prompt that asks for the first
                  invalid step in each; the challenger takes over only if it wins both
    resolve       a fresh high-effort solver shown only the candidate options
    verify        one call per answer: check it and its reasoning, VERDICT: CORRECT or
                  INCORRECT; the most voted CORRECT answer, or the plurality if none or all are

--probe asks the pairwise prompt about (right answer, most voted wrong answer) in both
orders on every split pool that holds the right answer: how often the model picks the
right one, and how much the order sways it.

Every new call is cached in <out>/<split>/calls.jsonl (keyed by question, replicate,
kind and the request's text), the judge's rounds in the round runner's own file there;
the recorded caches are read only. Running again costs nothing; --no-live only reports.

    python scripts/chooser_lab.py --split train --pools 3H,2H2L --probe --show summary
    python scripts/chooser_lab.py --split train --pools 3H,2H2L \\
        --choosers plurality,judge,judge_blind,pairwise,resolve,verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_debate_mcq as B  # noqa: E402
import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_space as P  # noqa: E402
import schema_fitness as SF  # noqa: E402
from gap_report import configure_like, plurality  # noqa: E402

ROOT = P.ROOT
POOLS = {"3H": ["solver_x3|high"], "4H": ["solver_x4|high"], "2H2L": ["solver_x2|high", "solver_x2|blind"]}
CHOOSERS = ("plurality", "judge", "judge_blind", "pairwise", "resolve", "verify")
JUDGE_ROUND = "judge|high"

PAIR_PROMPT = ("You are a Judge. Two responses to a hard graduate-level multiple-choice question reach "
               "different answers. Examine each response's reasoning against the exact wording of the "
               "question and find the first invalid step in each: a false fact, a wrong deduction, a "
               "miscalculation, a misread option or condition. How confident, long or well written a "
               "response is is not evidence that it is correct. Then decide which of the two answers is "
               "correct; you must choose one of them. ")
VERIFY_PROMPT = ("You are a Verifier. A participant has answered a hard graduate-level multiple-choice "
                 "question; their answer and reasoning are shown. Check the answer independently against "
                 "the exact wording of the question: test whether that option satisfies every requirement "
                 "the question states, and check the decisive facts, steps and calculations of the "
                 "reasoning. Confident or detailed reasoning is not evidence that it is correct. End with "
                 "exactly one line: 'VERDICT: CORRECT' if the answer is the correct option, or "
                 "'VERDICT: INCORRECT' if it is not.")
VERDICT_NUDGE = ("Based on your reasoning above, reply with only the line 'VERDICT: CORRECT' or "
                 "'VERDICT: INCORRECT'.")
_VERDICT = re.compile(r"VERDICT\s*[:\-]\s*(CORRECT|INCORRECT)\b", re.I)


# --- pools --------------------------------------------------------------------------------

def load_pools(runner, rows: dict, pool: str, reps=range(5)) -> list[dict]:
    """Every recorded pool of this type: its rounds (as the runner replays them) and speakers."""
    specs = [P.PLAN_ROUNDS[name] for name in POOLS[pool]]
    out = []
    for q, row in rows.items():
        prompts = B.question_prompts(row)
        n = len(row["options"])
        for rep in reps:
            rounds = []
            for i, spec in enumerate(specs):
                _, _, hit, _ = runner.lookup(q, specs[:i], spec, rep, prompts)
                if hit is None:
                    break
                rounds.append(hit)
            if len(rounds) < len(specs):
                continue
            speakers = [{"persona": p, "effort": spec.get("effort", D.DEFAULT_EFFORT), "text": text,
                         "letter": D.extract_letter(text, n)}
                        for spec, rnd in zip(specs, rounds) for p, text in rnd]
            letters = [s["letter"] for s in speakers if s["letter"]]
            plural, tied = plurality(letters)
            out.append({"q": q, "rep": rep, "pool": pool, "specs": specs, "rounds": rounds,
                        "speakers": speakers, "votes": Counter(letters), "cands": list(dict.fromkeys(letters)),
                        "plural": plural, "tied": tied, "right": row["answer_letter"], "n": n,
                        "prompts": prompts})
    return out


def load_recorded(runner, paths: list[Path]) -> None:
    """Read recorded round caches into a runner, read-only, as BudgetedRunner's
    read_only_caches are (first write wins, errored and malformed lines skipped), and with
    each recording's completion tokens, which that loader leaves out (the judge's cost)."""
    for path in paths:
        for line in Path(path).open(errors="replace"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or not M.well_formed(r) or r.get("error"):
                continue
            key = (r["q"], r["k"], r["r"])
            if key not in runner._cache:
                runner._cache[key] = r["responses"]
                if isinstance(r.get("usage"), list):
                    runner._tokens[key] = sum((u or {}).get("completion", 0) for u in r["usage"])


def is_split(pool: dict) -> bool:
    return len(pool["cands"]) >= 2


def ranked(pool: dict) -> list[str]:
    """The candidates by votes, ties to the first committed (the plurality first)."""
    first = {l: i for i, l in enumerate(pool["cands"])}
    return sorted(pool["cands"], key=lambda l: (-pool["votes"][l], first[l]))


def representative(pool: dict, letter: str) -> dict:
    return next(s for s in pool["speakers"] if s["letter"] == letter)


def argument(speaker: dict, show: str) -> str:
    """What a chooser reads of one candidate: the summary later speakers read, or all of it."""
    if show == "summary":
        return D._shown(speaker["text"])
    full, summary = D._split_summary(speaker["text"])
    return full.strip() + (f"\n\n[summary]\n{summary.strip()}" if summary else "")


# --- model calls --------------------------------------------------------------------------

class Caller:
    """The lab's own model calls, cached in one JSONL file: appended, resumed, errors not
    cached. One main call at the chooser's effort with the executor's reply room; if its
    answer cannot be read, one low-effort follow-up on the same conversation."""

    def __init__(self, path: Path, base_urls: str, model: str, api_key: str = "EMPTY", live: bool = True):
        self.path, self.model, self.live = Path(path), model, live
        self.cache: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.new_calls = self.errors = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.open(errors="replace"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.cache.setdefault(r["key"], r)
        self.fh = self.path.open("a")
        self.clients = None
        if live:
            from openai import OpenAI
            self.clients = cycle([OpenAI(base_url=u.strip(), api_key=api_key, timeout=D.REPLY_TIMEOUT, max_retries=2)
                                  for u in base_urls.split(",") if u.strip()])

    def key(self, q: str, rep: int, kind: str, system: str, user: str, effort: str) -> str:
        h = hashlib.sha1(json.dumps([self.model, system, user, effort]).encode()).hexdigest()[:16]
        return f"{q}|{rep}|{kind}|{h}"

    def _client(self):
        with self.lock:
            return next(self.clients)

    def ask(self, q: str, rep: int, kind: str, system: str, user: str, effort: str,
            parse, nudge: str, parse_followup=None) -> dict | None:
        """{"text", "followup", "value", "completion", "calls"}, or None when it could not be
        had (no live server, or the call failed twice). `parse(text)` reads the answer."""
        parse_followup = parse_followup or parse
        k = self.key(q, rep, kind, system, user, effort)
        with self.lock:
            rec = self.cache.get(k)
        if rec is None:
            if not self.live:
                return None
            rec = self._call(k, q, rep, kind, system, user, effort, parse, nudge)
            if rec is None:
                return None
        value = parse(rec["text"])
        if value is None and rec.get("followup"):
            value = parse_followup(rec["followup"])
        return {**rec, "value": value}

    def _call(self, k, q, rep, kind, system, user, effort, parse, nudge) -> dict | None:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(2):
            try:
                client = self._client()
                room = max(D.MIN_REPLY_TOKENS,
                           D.WINDOW - D._prompt_tokens(client, self.model, messages) - D.SUMMARY_RESERVE)
                resp = D._complete(client, self.model, messages, effort, room)
                break
            except Exception as exc:           # noqa: BLE001 -- retried once, then the pool waits for a rerun
                err = str(exc)
        else:
            with self.lock:
                self.errors += 1
            print(f"  call failed ({kind}, {q}, rep {rep}): {err[:200]}", file=sys.stderr)
            return None
        thinking, visible = D._parts(resp)
        usage = getattr(resp, "usage", None)
        rec = {"key": k, "q": q, "r": rep, "kind": kind, "effort": effort, "text": visible,
               "thinking_chars": len(thinking), "finish": getattr(resp.choices[0], "finish_reason", None),
               "completion": getattr(usage, "completion_tokens", 0) or 0,
               "prompt": getattr(usage, "prompt_tokens", 0) or 0, "calls": 1, "followup": None}
        if parse(visible) is None:
            hidden = bool(thinking) and (effort == "high" or len(visible) < D.VISIBLE_MIN_CHARS)
            tail = thinking[-D.THINKING_TAIL:] if hidden else ""
            shown = (f"[the end of my private reasoning]\n{tail}\n\n[my reply]\n{visible}" if tail
                     else visible) or "(no reply)"
            follow = messages + [{"role": "assistant", "content": shown}, {"role": "user", "content": nudge}]
            try:
                fresp = D._complete(self._client(), self.model, follow, "low", D.COMMIT_TOKENS_V3)
                rec["followup"] = D._parts(fresp)[1]
                rec["calls"] += 1
                rec["completion"] += getattr(getattr(fresp, "usage", None), "completion_tokens", 0) or 0
            except Exception:                  # noqa: BLE001 -- the main reply stands without a pick
                rec["followup"] = ""
        with self.lock:
            if k not in self.cache:
                self.cache[k] = rec
                self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.fh.flush()
            self.new_calls += rec["calls"]
            return self.cache[k]

    def close(self) -> None:
        self.fh.close()


def letter_parser(n: int, cands: list[str]):
    """(main, follow-up) readers of a pick among `cands`."""
    def main(text):
        l = D.extract_letter(text, n) or D.extract_letter((text or "").replace("*", ""), n)
        return l if l in cands else None

    def follow(text):
        l = D.extract_letter(text, n) or D.commit_letter_lenient(text, n)
        return l if l in cands else None
    return main, follow


def together(*thunks):
    """Run independent calls at once (both orders of a match, every answer of a verify)."""
    if len(thunks) == 1:
        return [thunks[0]()]
    with ThreadPoolExecutor(max_workers=len(thunks)) as pool:
        return list(pool.map(lambda f: f(), thunks))


def verdict(text: str | None) -> str | None:
    found = _VERDICT.findall((text or "").replace("*", ""))
    return found[-1].upper() if found else None


# --- choosers -------------------------------------------------------------------------------

class Lab:
    def __init__(self, rows: dict, caller: Caller, runner, show: str, live: bool):
        self.rows, self.caller, self.runner, self.show, self.live = rows, caller, runner, show, live

    def base(self, pool: dict) -> str:
        row = self.rows[pool["q"]]
        return D.render_question(row["question"], list(row["options"]))

    def pair(self, pool: dict, a: str, b: str) -> dict | None:
        """The pairwise prompt with `a` shown first."""
        ra, rb = representative(pool, a), representative(pool, b)
        system = PAIR_PROMPT + D.ANSWER_INSTR + " Always choose exactly one letter; never abstain."
        user = (f"{self.base(pool)}\n\nTwo responses to this question reach different answers.\n\n"
                f"Response 1 (answer {a}):\n{argument(ra, self.show)}\n\n"
                f"Response 2 (answer {b}):\n{argument(rb, self.show)}\n\n"
                f"Find the first invalid step in each response, then decide between {a} and {b}. "
                f"Give your response. {D.ANSWER_INSTR}")
        main, follow = letter_parser(pool["n"], [a, b])
        return self.caller.ask(pool["q"], pool["rep"], f"pair_{self.show}", system, user, "high",
                               main, D.judge_pick_nudge([a, b]), follow)

    def run(self, pool: dict, chooser: str) -> dict | None:
        """{"chosen", "completion", "calls"} or None if a call is missing."""
        if chooser == "plurality":
            return {"chosen": pool["plural"], "completion": 0, "calls": 0}
        if chooser == "judge":
            return self.judge(pool)
        if chooser == "judge_blind":
            return self.judge_blind(pool)
        if chooser == "pairwise":
            return self.pairwise(pool)
        if chooser == "resolve":
            return self.resolve(pool)
        if chooser == "verify":
            return self.verify(pool)
        raise ValueError(chooser)

    def judge(self, pool: dict) -> dict | None:
        spec = P.PLAN_ROUNDS[JUDGE_ROUND]
        try:
            out = self.runner.run_round(pool["q"], pool["rounds"], pool["specs"], spec, rep=pool["rep"],
                                        prompts=pool["prompts"])
        except M.OffCache:
            return None
        letters = SF.committed_letters(pool["rounds"] + [out], pool["n"])     # read as the seeds read
        tokens = self.runner.round_tokens(pool["q"], pool["specs"], spec, rep=pool["rep"], prompts=pool["prompts"])
        return {"chosen": letters[-1] if letters else None, "completion": tokens, "calls": 1}

    def judge_blind(self, pool: dict) -> dict | None:
        order = list(pool["cands"])
        random.Random(f"{pool['q']}|{pool['rep']}|{pool['pool']}").shuffle(order)
        shown = "\n\n".join(f"[Response {i + 1}] chose {a}:\n{argument(representative(pool, a), self.show)}"
                            for i, a in enumerate(order))
        user = (f"{self.base(pool)}\n\nPrior responses:\n{shown}\n\n{D.judge_block(sorted(pool['cands']))}"
                f"\n\nGive your response. {D.ANSWER_INSTR}")
        main, follow = letter_parser(pool["n"], pool["cands"])
        r = self.caller.ask(pool["q"], pool["rep"], f"judge_blind_{self.show}", D.PERSONA_PROMPTS[D.JUDGE_PERSONA],
                            user, "high", main, D.judge_pick_nudge(sorted(pool["cands"])), follow)
        if r is None:
            return None
        return {"chosen": r["value"] or pool["plural"], "completion": r["completion"], "calls": r["calls"],
                "no_pick": r["value"] is None}

    def pairwise(self, pool: dict) -> dict | None:
        order = ranked(pool)
        champ, completion, calls, matches = order[0], 0, 0, []
        for chal in order[1:]:
            first, second = together(lambda: self.pair(pool, champ, chal), lambda: self.pair(pool, chal, champ))
            if first is None or second is None:
                return None
            completion += first["completion"] + second["completion"]
            calls += first["calls"] + second["calls"]
            matches.append([champ, chal, first["value"], second["value"]])
            if first["value"] == chal and second["value"] == chal:
                champ = chal
        return {"chosen": champ, "completion": completion, "calls": calls, "matches": matches}

    def resolve(self, pool: dict) -> dict | None:
        row = self.rows[pool["q"]]
        keep = sorted(pool["cands"])
        body = "\n".join(f"{l}) {row['options'][D.LETTERS.index(l)]}" for l in keep)
        user = (f"{row['question']}\n\n{body}\n\nOnly these options are under consideration; choose one of them "
                f"({', '.join(keep)}).\n\n{D.ANSWER_INSTR}")
        main, follow = letter_parser(pool["n"], keep)
        r = self.caller.ask(pool["q"], pool["rep"], "resolve", D.PERSONA_PROMPTS["solver"], user, "high",
                            main, D.judge_pick_nudge(keep), follow)
        if r is None:
            return None
        return {"chosen": r["value"] or pool["plural"], "completion": r["completion"], "calls": r["calls"],
                "no_pick": r["value"] is None}

    def verify(self, pool: dict) -> dict | None:
        def one(a: str):
            user = (f"{self.base(pool)}\n\nThe participant answered {a}:\n"
                    f"{argument(representative(pool, a), self.show)}\n\n"
                    f"Give your response. End with the line 'VERDICT: CORRECT' or 'VERDICT: INCORRECT'.")
            return self.caller.ask(pool["q"], pool["rep"], f"verify_{self.show}", VERIFY_PROMPT, user, "high",
                                   verdict, VERDICT_NUDGE)

        got = together(*[lambda a=a: one(a) for a in pool["cands"]])
        if any(r is None for r in got):
            return None
        verdicts = {a: r["value"] for a, r in zip(pool["cands"], got)}
        completion, calls = sum(r["completion"] for r in got), sum(r["calls"] for r in got)
        valid = [a for a in ranked(pool) if verdicts[a] == "CORRECT"]
        chosen = valid[0] if 0 < len(valid) < len(pool["cands"]) else pool["plural"]
        return {"chosen": chosen, "completion": completion, "calls": calls, "verdicts": verdicts}

    def probe(self, pool: dict) -> dict | None:
        """(right, most voted wrong answer) in both orders."""
        wrong = next(l for l in ranked(pool) if l != pool["right"])
        first, second = together(lambda: self.pair(pool, pool["right"], wrong),
                                 lambda: self.pair(pool, wrong, pool["right"]))
        if first is None or second is None:
            return None
        return {"wrong": wrong, "right_first": first["value"], "right_second": second["value"],
                "completion": first["completion"] + second["completion"], "calls": first["calls"] + second["calls"],
                "effort_right": representative(pool, pool["right"])["effort"],
                "effort_wrong": representative(pool, wrong)["effort"],
                "votes_right": pool["votes"][pool["right"]], "votes_wrong": pool["votes"][wrong]}


# --- metrics --------------------------------------------------------------------------------

def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if not n:
        return float("nan"), float("nan")
    m = sum(xs) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else 0.0)


def chooser_stats(results: list[tuple[dict, dict]]) -> dict:
    """(pool, chooser result) pairs of split pools -> the chooser's numbers against the plurality."""
    n = len(results)
    if not n:
        return {"n": 0}
    ok = [int(r["chosen"] == p["right"]) for p, r in results]
    pl = [int(p["plural"] == p["right"]) for p, _ in results]
    diff_m, diff_se = mean_se([a - b for a, b in zip(ok, pl)])
    clear = [(p, r) for p, r in results if len(p["tied"]) == 1]
    over = [(p, r) for p, r in clear if r["chosen"] != p["plural"]]
    cf = sum(r["chosen"] == p["right"] for p, r in over)
    wf = sum(p["plural"] == p["right"] for p, r in over)
    ties = [(p, r) for p, r in results if len(p["tied"]) > 1]
    return {"n": n, "present": 100 * sum(p["right"] in p["cands"] for p, _ in results) / n,
            "acc": 100 * sum(ok) / n, "plurality": 100 * sum(pl) / n,
            "plurality_random": 100 * sum((p["right"] in p["tied"]) / len(p["tied"]) for p, _ in results) / n,
            "diff": 100 * diff_m, "diff_se": 100 * diff_se,
            "clear": len(clear), "overrides": len(over), "cf": cf, "wf": wf, "net": cf - wf,
            "precision": cf / (cf + wf) if cf + wf else None,
            "ties": len(ties), "tie_acc": 100 * sum(r["chosen"] == p["right"] for p, r in ties) / len(ties) if ties else None,
            "tie_random": 100 * sum((p["right"] in p["tied"]) / len(p["tied"]) for p, _ in ties) / len(ties) if ties else None,
            "calls": sum(r["calls"] for _, r in results) / n,
            "completion": sum(r["completion"] or 0 for _, r in results) / n,
            "no_pick": sum(bool(r.get("no_pick")) for _, r in results)}


def probe_stats(items: list[tuple[dict, dict]]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    a1 = [r["right_first"] == p["right"] for p, r in items]
    a2 = [r["right_second"] == p["right"] for p, r in items]
    # position: which response was picked, per call
    picks_first = sum(r["right_first"] == p["right"] for p, r in items) + sum(r["right_second"] == r["wrong"] for p, r in items)
    answered = sum(r["right_first"] is not None for _, r in items) + sum(r["right_second"] is not None for _, r in items)
    return {"n": n, "acc_right_first": 100 * sum(a1) / n, "acc_right_second": 100 * sum(a2) / n,
            "acc_mean": 50 * (sum(a1) + sum(a2)) / n, "both_right": 100 * sum(x and y for x, y in zip(a1, a2)) / n,
            "both_wrong": 100 * sum(not x and not y for x, y in zip(a1, a2)) / n,
            "split": 100 * sum(x != y for x, y in zip(a1, a2)) / n,
            "picks_response_1": 100 * picks_first / answered if answered else None,
            "no_pick": 2 * n - answered,
            "calls": sum(r["calls"] for _, r in items) / n, "completion": sum(r["completion"] for _, r in items) / n}


def f1(x) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def chooser_table(rows: list[tuple[str, dict]]) -> list[str]:
    out = ["| chooser | split pools | right present | chooser | plurality (random tie) | chooser - plurality "
           "| overrides of a clear plurality | CF | WF | precision | net | ties: chooser vs random | calls "
           "| completion tokens |", "|---|" + "---:|" * 13]
    for name, s in rows:
        if not s.get("n"):
            out.append(f"| {name} | 0 |" + " - |" * 12)
            continue
        out.append(f"| {name} | {s['n']} | {f1(s['present'])} | {f1(s['acc'])} | {f1(s['plurality'])} "
                   f"({f1(s['plurality_random'])}) | {s['diff']:+.1f} ± {s['diff_se']:.1f} | {s['overrides']} of "
                   f"{s['clear']} | {s['cf']} | {s['wf']} | "
                   + (f"{100 * s['precision']:.0f}%" if s["precision"] is not None else "-")
                   + f" | {s['net']:+d} | {f1(s['tie_acc'])} vs {f1(s['tie_random'])} ({s['ties']}) | "
                   f"{s['calls']:.1f} | {s['completion'] / 1000:.1f}k |")
    return out


def probe_table(rows: list[tuple[str, dict]]) -> list[str]:
    out = ["| pairs | n | right shown first | right shown second | mean | both orders right | both wrong "
           "| orders disagree | picks response 1 | calls | completion tokens |", "|---|" + "---:|" * 10]
    for name, s in rows:
        if not s.get("n"):
            continue
        out.append(f"| {name} | {s['n']} | {f1(s['acc_right_first'])} | {f1(s['acc_right_second'])} | "
                   f"{f1(s['acc_mean'])} | {f1(s['both_right'])} | {f1(s['both_wrong'])} | {f1(s['split'])} | "
                   f"{f1(s['picks_response_1'])} | {s['calls']:.1f} | {s['completion'] / 1000:.1f}k |")
    return out


# --- main -----------------------------------------------------------------------------------

def run_jobs(jobs, fn, workers: int, desc: str) -> list:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(tqdm(pool.map(fn, jobs), total=len(jobs), desc=desc, unit="job", disable=len(jobs) < 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=ROOT / "outputs/pipeline_cluster_gptoss/run3",
                    help="the search run whose executor settings (archive header) and caches are used")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--pools", default="3H,2H2L", help=f"pool types, of {', '.join(POOLS)}")
    ap.add_argument("--choosers", default="", help=f"of {', '.join(CHOOSERS)}")
    ap.add_argument("--probe", action="store_true", help="the pairwise probe on split pools that hold the right answer")
    ap.add_argument("--show", choices=["summary", "full"], default="summary",
                    help="what choosers read of each candidate (the judge always reads the debate as speakers do)")
    ap.add_argument("--limit", type=int, default=None, help="at most this many split pools per type (a fixed sample)")
    ap.add_argument("--caches", default=None, help="recorded round caches, comma-separated (default: by --split)")
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default outputs/chooser_lab/<model tag>")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--no-live", action="store_true", help="cache only: report what is recorded")
    args = ap.parse_args()

    header = json.loads((args.run / "archive.jsonl").open().readline())
    settings = header["settings"]
    configure_like(settings)
    if JUDGE_ROUND not in P.PLAN_ROUNDS:
        raise SystemExit("the run's settings have no judge persona; the judge chooser needs it")
    model, tag = settings["model"], P.model_tag(settings["model"])
    pools_wanted = [p for p in args.pools.split(",") if p]
    choosers = [c for c in args.choosers.split(",") if c]
    for c in choosers:
        if c not in CHOOSERS:
            raise SystemExit(f"unknown chooser {c!r}; choose from {CHOOSERS}")
    if not choosers and not args.probe:
        raise SystemExit("nothing to do: give --choosers and/or --probe")
    if args.split == "train":
        dataset = args.dataset or ROOT / "datasets/supergpqa_600_train.json"
        caches = [args.run.parent / f"rounds_{tag}.jsonl"]
        group_of = {q: int(g) for g, qs in header["groups"].items() for q in qs}
    else:
        dataset = args.dataset or ROOT / "datasets/supergpqa_600_test.json"
        caches = [args.run / "test_eval" / f"rounds_{tag}.jsonl"]
        res = json.loads(sorted((args.run / "test_eval").glob("results_k*.json"))[-1].read_text())
        group_of = {q: r["group"] for q, r in json.loads((ROOT / res["routes"]).read_text())["routes"].items()}
    if args.caches:
        caches = [Path(c) for c in args.caches.split(",")]
    rows = {r["id"]: r for r in json.loads(dataset.read_text())}
    out_dir = (args.out or ROOT / "outputs/chooser_lab" / tag) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    live = not args.no_live
    runner = M.BudgetedRunner(rows, [], lock=live, base_urls=args.base_urls, model=model, temperature=1.0,
                              answer_tokens=P.ANSWER_TOKENS, cache_path=out_dir / f"rounds_{tag}.jsonl",
                              max_calls=None, api_key=args.api_key, progress=False)
    load_recorded(runner, caches)
    runner.reset_budget(None if live else 0)
    caller = Caller(out_dir / "calls.jsonl", args.base_urls, model, args.api_key, live=live)
    lab = Lab(rows, caller, runner, args.show, live)
    report: list[str] = [f"# Chooser lab: {args.split} pools, arguments shown as {args.show}", "",
                         f"settings {settings}; recorded caches {', '.join(map(str, caches))}"
                         + (f"; a fixed sample of at most {args.limit} split pools per type" if args.limit else ""),
                         ""]
    saved: dict = {"split": args.split, "show": args.show, "limit": args.limit, "pools": {}}
    try:
        for ptype in pools_wanted:
            pools = load_pools(runner, rows, ptype)
            split = [p for p in pools if is_split(p)]
            random.Random(0).shuffle(split)
            split = sorted(split[:args.limit] if args.limit else split, key=lambda p: (p["q"], p["rep"]))
            unanimous_right = sum(p["plural"] == p["right"] for p in pools if not is_split(p))
            print(f"{ptype}: {len(pools)} pools, {sum(map(is_split, pools))} split; using {len(split)}")
            report += [f"## {ptype}: {len(pools)} pools, {sum(map(is_split, pools))} split"
                       + (f" ({len(split)} used)" if len(split) < sum(map(is_split, pools)) else ""), ""]
            entry: dict = {"n_pools": len(pools), "n_split": sum(map(is_split, pools)), "used": len(split)}
            if args.probe:
                cand = [p for p in split if p["right"] in p["cands"]]
                got = run_jobs(cand, lab.probe, args.workers, f"{ptype} probe")
                items = [(p, r) for p, r in zip(cand, got) if r is not None]
                rows_ = [("all", probe_stats(items))]
                for label, key in (("pairing (right, wrong)", lambda p, r: f"{r['effort_right']}-{r['effort_wrong']}"),
                                   ("votes", lambda p, r: "right has more votes" if r["votes_right"] > r["votes_wrong"]
                                    else "equal votes" if r["votes_right"] == r["votes_wrong"] else "right has fewer votes"),
                                   ("group", lambda p, r: f"group {group_of.get(p['q'])}")):
                    sub = defaultdict(list)
                    for p, r in items:
                        sub[key(p, r)].append((p, r))
                    rows_ += [(f"{label}: {k}", probe_stats(v)) for k, v in sorted(sub.items())]
                report += [f"### Pairwise probe ({len(items)} of {len(cand)} pairs answered)", ""]
                report += probe_table(rows_) + [""]
                entry["probe"] = {"stats": dict(rows_), "items": [
                    {"q": p["q"], "rep": p["rep"], "right": p["right"], **r} for p, r in items]}
            if choosers:
                stats, per_pool = {}, defaultdict(dict)
                for c in choosers:
                    got = run_jobs(split, lambda p, c=c: lab.run(p, c), args.workers, f"{ptype} {c}")
                    res_ = [(p, r) for p, r in zip(split, got) if r is not None]
                    stats[c] = {"all": chooser_stats(res_)}
                    by_g = defaultdict(list)
                    for p, r in res_:
                        by_g[group_of.get(p["q"])].append((p, r))
                        per_pool[f"{p['q']}|{p['rep']}"][c] = r
                    stats[c]["by_group"] = {str(g): chooser_stats(v) for g, v in sorted(by_g.items(), key=str)}
                    if len(res_) == len(split) == entry["n_split"]:
                        stats[c]["whole"] = 100 * (unanimous_right + sum(r["chosen"] == p["right"] for p, r in res_)) \
                            / len(pools)
                report += ["### Choosers on the split pools", ""]
                report += chooser_table([(c, stats[c]["all"]) for c in choosers]) + [""]
                whole = [(c, stats[c]["whole"]) for c in choosers if "whole" in stats[c]]
                if whole:
                    report += ["Accuracy over all pools of this type (unanimous pools keep their answer): "
                               + ", ".join(f"{c} {v:.1f}" for c, v in whole), ""]
                for g in sorted({g for c in choosers for g in stats[c]["by_group"]}):
                    report += [f"#### group {g}", ""]
                    report += chooser_table([(c, stats[c]["by_group"].get(g, {"n": 0})) for c in choosers]) + [""]
                entry["choosers"] = stats
                entry["per_pool"] = {k: {c: {kk: vv for kk, vv in r.items()} for c, r in v.items()}
                                     for k, v in per_pool.items()}
                for k, v in entry["per_pool"].items():
                    q, rep = k.split("|")
                    p = next(x for x in split if x["q"] == q and x["rep"] == int(rep))
                    v["_pool"] = {"right": p["right"], "cands": p["cands"], "votes": dict(p["votes"]),
                                  "plural": p["plural"], "group": group_of.get(q)}
            saved["pools"][ptype] = entry
    finally:
        runner.close()
        caller.close()
    stem = ("probe_" if args.probe and not choosers else "report_") + args.show
    (out_dir / f"{stem}.md").write_text("\n".join(report) + "\n")
    (out_dir / f"{stem}.json").write_text(json.dumps(saved, indent=1, default=str))
    print("\n".join(report))
    print(f"\nnew model calls {caller.new_calls}, failed {caller.errors}, judge rounds run {runner.novel_calls}; "
          f"wrote {out_dir / (stem + '.md')} and .json")


if __name__ == "__main__":
    main()

"""Shared fitness engine for POPULATION-level schema evolution.

The existing loop (evolve_debate_mcq.py) evaluates one schema on one question with
one execution -- a Bernoulli sample at a 3.9% base rate, which cannot rank two
schemas. Every design in evol_debate_designs.md instead scores a schema by its
solve rate over a SHARED batch of questions. This module is that scoring engine,
shared by evolve_cem_mcq.py (E5) and mapelites_mcq.py (E3).

Four things it provides:

  NESTED PAIRED BATCHES. One seeded permutation of the pool; stage k evaluates on
  order[:n_k]. Because the batches are prefixes of each other, (a) every candidate
  is scored on the SAME questions -- paired comparison, the variance reduction that
  makes 2-3pp differences resolvable -- and (b) promoting a candidate from a 64- to
  a 256-question stage costs only the incremental 192, since the first 64 are
  already cached.

  A PERSISTENT EXECUTION CACHE keyed (question, canonical schema, replicate). Runs
  resume after a kill, identical schemas proposed twice are free, and re-evaluating
  an archive elite draws a genuinely fresh sample by incrementing the replicate.

  THE DEPARTURE DECOMPOSITION, computed on every execution. accuracy = D x P, where
  D = P(final letter != the base model's letter) and P = P(correct | departed). The
  base letters come from the cached best-of-n run, so D costs nothing extra and --
  unlike accuracy -- needs no gold label. That makes it usable as a cheap first
  stage in a racing cascade, and as a behavior descriptor for MAP-Elites.

  A CALL COUNTER. Every reported number in this project has to be per-LLM-call, not
  per-generation, or the comparison against best-of-k is meaningless. The evaluator
  counts calls and enforces a hard budget.

Not a script -- import it.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path

from scipy.stats import beta as beta_dist
from tqdm import tqdm

import debate_mcq as D


def log(msg: str = "") -> None:
    """Print without tearing the progress bar. Safe when no bar is active."""
    tqdm.write(str(msg))


# --- schema helpers --------------------------------------------------------

def canon(schema: dict, prompts: dict | None = None) -> str:
    """Stable cache/identity key. Persona order within a round is meaningful
    (each persona sees the previous ones), so it is NOT sorted.

    A round's `sees` is emitted ONLY when it differs from the default, and the
    prompt hash only when prompts are overridden -- so keys minted before the E1
    genome extension still match, and old execution caches stay valid."""
    rounds = []
    for r in schema["rounds"]:
        d = {"personas": list(r["personas"])}
        if r.get("sees", D.DEFAULT_SEES) != D.DEFAULT_SEES:
            d["sees"] = r["sees"]
        rounds.append(d)
    key = {"rounds": rounds, "final": schema["final"]}
    if prompts:
        blob = json.dumps({k: prompts[k] for k in sorted(prompts)}, separators=(",", ":"))
        key["p"] = hashlib.sha1(blob.encode()).hexdigest()[:12]
    return json.dumps(key, sort_keys=False, separators=(",", ":"))


def n_calls(schema: dict) -> int:
    """LLM calls one execution costs: one per persona."""
    return sum(len(r["personas"]) for r in schema["rounds"])


_SEES_TAG = {"all": "", "none": "|blind", "letters_only": "|letters",
             "no_letters": "|nocommit", "last_round": "|lastrnd"}


def gloss(schema: dict) -> str:
    parts = ["+".join(r["personas"]) + _SEES_TAG.get(r.get("sees", D.DEFAULT_SEES), "")
             for r in schema["rounds"]]
    return "; ".join(parts) + f" -> {schema['final']}"


# --- statistics ------------------------------------------------------------

def beta_lcb(k: int, n: int, q: float = 0.25) -> float:
    """Lower confidence bound on a success rate: the q-quantile of the Beta
    posterior under a Jeffreys prior. Ranking by this instead of by k/n is what
    stops one lucky execution from holding an elite slot forever."""
    if n <= 0:
        return 0.0
    return float(beta_dist.ppf(q, k + 0.5, n - k + 0.5))


def summarize(outcomes: list[dict]) -> dict:
    """Fitness record for one schema on one batch, departure-decomposed."""
    n = len(outcomes)
    if not n:
        return {"n": 0, "n_correct": 0, "accuracy": 0.0, "lcb": 0.0, "n_departed": 0,
                "n_dep_correct": 0, "departure_rate": 0.0, "departure_precision": 0.0}
    k = sum(o["correct"] for o in outcomes)
    dep = [o for o in outcomes if o["departed"]]
    dep_hit = sum(o["correct"] for o in dep)
    return {"n": n, "n_correct": k, "accuracy": k / n, "lcb": beta_lcb(k, n),
            "n_departed": len(dep), "n_dep_correct": dep_hit, "departure_rate": len(dep) / n,
            "departure_precision": (dep_hit / len(dep)) if dep else 0.0}


# --- question pool ---------------------------------------------------------

class Pool:
    """A seeded permutation of the question pool, exposed as nested prefixes."""

    def __init__(self, dataset: Path, base_letters: dict[str, str], seed: int = 0,
                 limit: int | None = None):
        rows = json.loads(Path(dataset).read_text())
        rows = [r for r in rows if r["id"] in base_letters]   # need a base letter for D
        if limit:
            rows = rows[:limit]
        if not rows:
            raise SystemExit(f"no usable questions in {dataset} (missing base letters?)")
        self.rows = {r["id"]: r for r in rows}
        self.order = [r["id"] for r in rows]
        random.Random(seed).shuffle(self.order)
        self.base = base_letters

    def batch(self, n: int) -> list[str]:
        return self.order[: min(n, len(self.order))]

    def __len__(self) -> int:
        return len(self.order)


def load_base_letters(path: Path) -> dict[str, str]:
    """{question id -> the base model's answer}, as the modal letter of the cached
    best-of-n minimal-schema samples. Label-free."""
    out = {}
    for line in Path(path).open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("majority"):
            out[r["id"]] = r["majority"]
    return out


# --- evaluator -------------------------------------------------------------

class BudgetExhausted(RuntimeError):
    pass


class Evaluator:
    """Executes schemas over question batches, with caching, call accounting and
    a hard budget. Thread-safe."""

    def __init__(self, pool: Pool, base_urls: str, model: str, temperature: float,
                 answer_tokens: int, workers: int, cache_path: Path,
                 max_calls: int | None = None, api_key: str = "EMPTY",
                 progress: bool = True, request_timeout: float | None = 900.0):
        from openai import OpenAI
        urls = [u.strip() for u in base_urls.split(",") if u.strip()]
        self._clients = cycle([OpenAI(base_url=u, api_key=api_key, timeout=request_timeout,
                                      max_retries=2) for u in urls])
        self.pool, self.model = pool, model
        self.temperature, self.answer_tokens, self.workers = temperature, answer_tokens, workers
        self.max_calls = max_calls
        self.calls = 0
        self.executions = 0
        self.errors = 0
        self._lock = threading.Lock()
        self._cache: dict[tuple, dict] = {}
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():                      # resume
            bad = 0
            for line in self.cache_path.open(errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:                                      # tolerate a torn line from
                    r = json.loads(line)                  # an interrupted write
                except json.JSONDecodeError:
                    bad += 1
                    continue
                self._cache[(r["q"], r["s"], r["r"])] = r
            log(f"  resumed {len(self._cache)} cached executions from {self.cache_path}"
                + (f" ({bad} unreadable lines skipped)" if bad else ""))
        self._fh = self.cache_path.open("a")
        # One persistent bar for the whole run, advanced per completed execution. These
        # searches evaluate one candidate at a time and report only every N candidates,
        # so without this the first several thousand calls look like a hang.
        self._bar = tqdm(total=max_calls, unit="call", desc="starting", dynamic_ncols=True,
                         smoothing=0.02, miniters=1) if progress else None

    def stage(self, label: str) -> None:
        if self._bar is not None:
            self._bar.set_description(label[:48])

    def tick(self, cost: int) -> None:
        """Account for an execution issued OUTSIDE `run()` -- E7 replays failed runs
        directly to capture transcripts. Without this the bar silently under-reports
        and stalls during every reflection phase."""
        with self._lock:
            self.executions += 1
            if self._bar is not None:
                self._bar.update(cost)

    # -- internals
    def _client(self):
        with self._lock:
            return next(self._clients)

    def _charge(self, cost: int) -> None:
        with self._lock:
            if self.max_calls is not None and self.calls + cost > self.max_calls:
                raise BudgetExhausted(f"call budget {self.max_calls} exhausted at {self.calls}")
            self.calls += cost

    def _execute(self, qid: str, schema: dict, key: tuple, prompts: dict | None = None) -> dict:
        row = self.pool.rows[qid]
        cost = n_calls(schema)
        self._charge(cost)
        letter, err = None, None
        for _ in range(2):                                # one retry on a transport hiccup
            try:
                letter = D.execute_schema(self._client(), self.model, row["question"],
                                          list(row["options"]), schema,
                                          self.temperature, self.answer_tokens, prompts=prompts)
                err = None
                break
            except Exception as exc:
                err = str(exc)
        rec = {"q": qid, "s": key[1], "r": key[2], "letter": letter,
               "correct": (letter is not None and letter == row["answer_letter"]),
               "departed": (letter is not None and letter != self.pool.base.get(qid)),
               "error": err}
        with self._lock:
            self.executions += 1
            if err:
                self.errors += 1
            self._cache[key] = rec
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            if self._bar is not None:
                self._bar.update(cost)
                if self.errors:
                    self._bar.set_postfix(err=self.errors, refresh=False)
        return rec

    # -- public
    def run(self, schema: dict, qids: list[str], rep: int = 0,
            prompts: dict | None = None) -> list[dict]:
        """Outcomes for `schema` on `qids`. Cached (question, genome, rep) triples
        are free; only the missing ones are executed. Raising `rep` draws a fresh
        independent sample -- that is how an archive elite gets re-evaluated.
        `prompts` participates in the cache key, so E7's individuals never collide."""
        c = canon(schema, prompts)
        missing = [q for q in qids if (q, c, rep) not in self._cache]
        if missing:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(lambda q: self._execute(q, schema, (q, c, rep), prompts), missing))
        return [self._cache[(q, c, rep)] for q in qids if (q, c, rep) in self._cache]

    def evaluate(self, schema: dict, qids: list[str], rep: int = 0,
                 prompts: dict | None = None) -> dict:
        return summarize(self.run(schema, qids, rep, prompts))

    def close(self) -> None:
        """Close the bar and the cache handle. Call this BEFORE printing a final
        report, so the report is not interleaved with a live bar."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        if not self._fh.closed:
            self._fh.close()


# --- racing cascade --------------------------------------------------------

def race(schemas: list[dict], ev: Evaluator, stages: tuple[int, ...] = (64, 256, 1024),
         keep_frac: float = 0.5, dep_band: tuple[float, float] | None = None,
         rep: int = 0, verbose: bool = True) -> list[tuple[dict, dict]]:
    """Successive halving over nested prefix batches. Returns the survivors as
    (schema, stats) sorted by LCB, descending.

    Stage 0 optionally applies `dep_band` -- a LABEL-FREE filter on departure rate.
    A schema outside the band is killed before any gold label is consulted, which
    is the cheap first rung of the cascade.
    """
    alive = {canon(s): s for s in schemas}
    stats: dict[str, dict] = {}
    for si, n in enumerate(stages):
        if not alive:
            break
        qids = ev.pool.batch(n)
        for ci, (c, s) in enumerate(list(alive.items()), 1):
            ev.stage(f"stage{si} n={n} cand {ci}/{len(alive)}")
            stats[c] = ev.evaluate(s, qids, rep)
        if si == 0 and dep_band is not None:
            lo, hi = dep_band
            kept = {c: s for c, s in alive.items() if lo <= stats[c]["departure_rate"] <= hi}
            if kept:                                       # never empty the field
                alive = kept
        if si < len(stages) - 1:
            k = max(1, math.ceil(len(alive) * keep_frac))
            best = sorted(alive, key=lambda c: -stats[c]["lcb"])[:k]
            alive = {c: alive[c] for c in best}
        if verbose:
            top = max(alive, key=lambda c: stats[c]["lcb"])
            log(f"    stage {si} (n={n}): {len(alive):>3} alive | best acc "
                f"{stats[top]['accuracy']:.1%} lcb {stats[top]['lcb']:.1%} | "
                f"calls {ev.calls}")
    return sorted(((alive[c], stats[c]) for c in alive), key=lambda t: -t[1]["lcb"])


# --- round-level execution (per-question methods) ---------------------------
# Machinery for treegrow_mcq.py and adaptive_debate_mcq.py, which grow or trim a
# schema per question while it runs and therefore need (a) transcripts, not just
# final letters, and (b) a cache keyed by the PREFIX of rounds executed so far,
# so tree branches share their parent's rounds for free and a trimmed run of a
# recipe is a cache prefix of the full one.

# The per-question critic. Until 2026-10-06 it was the critic evolved by E7
# (outputs/evolve_persona_text_summary.json, individual i019: "Your job is to challenge the
# leading answer, not to restate it. ..."). Since then it is debate_mcq.CRITIC_BODY, which reads
# the whole discussion, not only the leading answer; it may keep the answer.
_CRITIC_BODY = D.CRITIC_BODY
REWRITTEN_CRITIC_V1 = (_CRITIC_BODY + "Reason step by step, then end with exactly one "
                       "line: 'ANSWER: <letter>'. Always choose exactly one letter; never abstain.")
# v2: the same critic, with the careful-reasoning instruction the other personas get.
REWRITTEN_CRITIC_V2 = (_CRITIC_BODY + D.ANSWER_INSTR_V2
                       + " Always choose exactly one letter; never abstain.")
REWRITTEN_CRITIC = REWRITTEN_CRITIC_V1
# open answers (debate_mcq.set_open_answers): the same critic, asked for an answer, not an option
_CRITIC_BODY_OPEN = D.CRITIC_BODY_OPEN


def set_v2(on: bool) -> None:
    """Switch the whole executor to v2 (careful-reasoning prompts, long replies,
    summary follow-up) or back. Entry scripts call THIS, not D.set_v2, so the
    per-question critic override changes with the persona prompts. (debate_mcq
    cannot import this module, so the critic text is switched here.)"""
    D.set_v2(on)
    _rebuild_critic()


def _rebuild_critic() -> None:
    """The per-question critic override from debate_mcq's current switches.
    With visible reasoning off this is exactly what set_v2 always chose."""
    global REWRITTEN_CRITIC
    if D.OPEN:          # D.ANSWER_INSTR carries the answer format and, if on, the visible sentence
        REWRITTEN_CRITIC = _CRITIC_BODY_OPEN + D.ANSWER_INSTR + " Always give exactly one final answer; never abstain."
        return
    if D.PLAIN_INSTR:
        REWRITTEN_CRITIC = _CRITIC_BODY + D.ANSWER_INSTR_PLAIN + " Always choose exactly one letter; never abstain."
    else:
        REWRITTEN_CRITIC = REWRITTEN_CRITIC_V2 if D.V2 else REWRITTEN_CRITIC_V1
    if D.VISIBLE_REASONING:
        REWRITTEN_CRITIC = REWRITTEN_CRITIC + D.VISIBLE_SENTENCE


def set_visible_reasoning(on: bool, detailed: bool = False) -> None:
    """Ask every persona to put its reasoning in the visible reply (for
    models that think in a hidden channel). Off by default; see debate_mcq."""
    D.set_visible_reasoning(on, detailed)
    _rebuild_critic()


def set_plain_instruction(on: bool) -> None:
    """The plain commitment instruction (debate_mcq.set_plain_instruction), with the
    critic override rebuilt."""
    D.set_plain_instruction(on)
    _rebuild_critic()


def set_v3(on: bool, window: int | None = None) -> None:
    """The v3 executor (debate_mcq.set_v3), with the critic override rebuilt."""
    D.set_v3(on, window)
    _rebuild_critic()


def set_open_answers(on: bool) -> None:
    """Open answers (debate_mcq.set_open_answers), with the critic override rebuilt."""
    D.set_open_answers(on)
    _rebuild_critic()


def set_math_answers(on: bool) -> None:
    """MATH answers (debate_mcq.set_math_answers; with open answers on), with the critic override rebuilt."""
    D.set_math_answers(on)
    _rebuild_critic()


def committed_letters(all_rounds: list[list[tuple[str, str]]], n: int) -> list[str]:
    """Ordered committed letters over a transcript: answering personas only,
    unparseable responses skipped."""
    out = []
    for rnd in all_rounds:
        for persona, resp in rnd:
            if persona in D.NON_ANSWERING:
                continue
            if (l := D.extract_letter(resp, n)):
                out.append(l)
    return out


def path_key(rounds: list[dict], prompts: dict | None = None) -> str:
    """Cache identity of a PARTIAL schema: the list of rounds executed so far.
    Same normalization as canon() -- `sees` emitted only when non-default, prompt
    overrides as a hash -- so equal prefixes collide and get reused. Under v3 a
    round's effort is emitted when not the default, and the first round's `sees`
    is dropped (it has nothing to see either way)."""
    norm = []
    for i, r in enumerate(rounds):
        d = {"personas": list(r["personas"])}
        if r.get("sees", D.DEFAULT_SEES) != D.DEFAULT_SEES and not (D.V3 and i == 0):
            d["sees"] = r["sees"]
        if r.get("effort", D.DEFAULT_EFFORT) != D.DEFAULT_EFFORT:
            d["effort"] = r["effort"]
        norm.append(d)
    return _finish_key({"rounds": norm}, prompts)


def blind_key(persona: str, effort: str, ordinal: int, prompts: dict | None = None) -> str:
    """Cache identity of ONE blind speaker under v3: a speaker who sees only the
    question, so its reply does not depend on the debate before it. `ordinal`
    counts the earlier blind calls of the same persona and effort in the debate,
    so the k-th blind solver of every program is the same sample: four blind
    solvers in one program are the first two of another plus a later pair."""
    return _finish_key({"blind": [persona, effort, ordinal]}, prompts)


def _finish_key(key: dict, prompts: dict | None) -> str:
    """The settings every key carries after its round description."""
    if prompts:
        blob = json.dumps({k: prompts[k] for k in sorted(prompts)}, separators=(",", ":"))
        key["p"] = hashlib.sha1(blob.encode()).hexdigest()[:12]
    # How much of each prior response a persona was shown changes what the
    # model saw, so it has to change the key too -- otherwise recordings made
    # under a different window replay as if they were the same run. Emitted
    # only when non-default, so every existing recording keeps its key.
    if (sig := D.digest_signature()) is not None:
        key["d"] = list(sig)
    # Same for the commit follow-up: a reply that was nudged to commit is a
    # different recording from one that was left truncated.
    if (c := D.commit_signature()) is not None:
        key["c"] = c
    # v2 recordings (careful-reasoning prompts, long replies, summary appended)
    # are a different run from v1 ones. Appended last so v1 keys are unchanged.
    if (v := D.v2_signature()) is not None:
        key["v"] = v
    # Visible reasoning changes every prompt. Appended last, and only when on,
    # so every existing key is unchanged.
    if (g := D.visible_signature()) is not None:
        key["g"] = g
    # The summary is what later speakers read, so its length limit changes what
    # they saw. Appended last, and only when not the original 120 words.
    if (w := D.summary_signature()) is not None:
        key["s"] = w
    # Under v3 a reply may fill the window, so the window is part of what it is.
    if (win := D.window_signature()) is not None:
        key["w"] = win
    # Open answers change every prompt. Appended last, and only when on.
    if (o := D.open_signature()) is not None:
        key["o"] = o
    # The plain instruction changes every prompt. Appended last, and only when on.
    if (i := D.plain_signature()) is not None:
        key["i"] = i
    # Every text a speaker is sent (persona prompts, the solver's question prompt, the discussion
    # format, the follow-ups), as a short hash: a change to any of them gives new recordings and
    # never replays old ones. In every key since 2026-10-06 (the prompts were rewritten then, so
    # no earlier recording replays under the current prompts).
    key["t"] = D.prompt_signature()
    return json.dumps(key, sort_keys=False, separators=(",", ":"))


def is_blind_round(executed_specs: list[dict], round_spec: dict) -> bool:
    """Under v3, a round whose speakers see nothing of the debate: the first
    round, or any round with sees 'none'. Its speakers are cached one by one."""
    return D.V3 and (not executed_specs or round_spec.get("sees", D.DEFAULT_SEES) == "none")


def round_keys(executed_specs: list[dict], round_spec: dict,
               prompts: dict | None = None) -> tuple[bool, list[str]]:
    """(per_speaker, keys) for running `round_spec` after `executed_specs`. A
    blind round has one key per speaker (blind_key, with the ordinals counted
    over the earlier blind rounds); any other round has one key, its path."""
    if not is_blind_round(executed_specs, round_spec):
        return False, [path_key(executed_specs + [round_spec], prompts)]
    seen: dict[tuple[str, str], int] = {}
    for i, spec in enumerate(executed_specs):
        if is_blind_round(executed_specs[:i], spec):
            eff = spec.get("effort", D.DEFAULT_EFFORT)
            for p in spec["personas"]:
                seen[(p, eff)] = seen.get((p, eff), 0) + 1
    eff = round_spec.get("effort", D.DEFAULT_EFFORT)
    keys = []
    for p in round_spec["personas"]:
        k = seen.get((p, eff), 0)
        seen[(p, eff)] = k + 1
        keys.append(blind_key(p, eff, k, prompts))
    return True, keys


class RoundRunner:
    """Executes schemas ONE ROUND at a time, with a transcript-level cache.

    Where Evaluator caches (question, whole schema, rep) -> final letter, this
    caches (question, rounds-executed-so-far, rep) -> the newest round's raw
    responses. Errored rounds are written to the file for diagnostics but never
    enter the cache, so a transient failure is retried on the next run instead
    of poisoning every descendant of that transcript. Thread-safe; the intended
    pattern is one thread per question."""

    def __init__(self, rows: dict[str, dict], base_urls: str, model: str, temperature: float,
                 answer_tokens: int, cache_path: Path, max_calls: int | None = None,
                 api_key: str = "EMPTY", progress: bool = True,
                 request_timeout: float | None = 900.0):
        from openai import OpenAI
        urls = [u.strip() for u in base_urls.split(",") if u.strip()]
        self._clients = cycle([OpenAI(base_url=u, api_key=api_key, timeout=request_timeout,
                                      max_retries=2) for u in urls])
        self.rows, self.model = rows, model
        self.temperature, self.answer_tokens = temperature, answer_tokens
        self.max_calls = max_calls
        self.calls = 0
        self.rounds_run = 0
        self.errors = 0
        self.followups = 0
        self.summaries = 0
        self._lock = threading.Lock()
        self._cache: dict[tuple, list] = {}
        self._tokens: dict[tuple, int] = {}               # completion tokens per recording, for reports
        # of those, the tokens of answer-locked summaries (recorded with debate_mcq.SUMMARY_SPLIT on;
        # absent from other recordings, whose tokens then all count)
        self._summary_tokens: dict[tuple, int] = {}
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():                      # resume; errored rows skipped
            bad = 0
            for line in self.cache_path.open(errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:                                      # a disk-full crash can leave
                    r = json.loads(line)                  # a torn or zero-filled line;
                except json.JSONDecodeError:              # that round simply re-runs
                    bad += 1
                    continue
                if not r.get("error"):
                    key = (r["q"], r["k"], r["r"])
                    self._cache[key] = r["responses"]
                    if isinstance(r.get("usage"), list):
                        self._tokens[key] = sum((u or {}).get("completion", 0) for u in r["usage"])
                        self._summary_tokens[key] = sum((u or {}).get("summary", 0) for u in r["usage"])
            log(f"  resumed {len(self._cache)} cached rounds from {self.cache_path}"
                + (f" ({bad} unreadable lines skipped)" if bad else ""))
        if self.cache_path.exists() and self.cache_path.stat().st_size:
            with self.cache_path.open("rb") as fh:          # a torn last line (a kill during a write)
                fh.seek(-1, 2)                              # must not swallow the next recording
                torn = fh.read(1) != b"\n"
            if torn:
                with self.cache_path.open("a") as fh:
                    fh.write("\n")
        self._fh = self.cache_path.open("a")
        # No total: max_calls is a safety rail, not a workload estimate, and using
        # it as the denominator shows a meaningless 0%-of-2M bar. Question-level
        # progress lives in the stage() label; the bar shows calls done and rate.
        self._bar = tqdm(total=None, unit="call", desc="starting", dynamic_ncols=True,
                         smoothing=0.02, miniters=1) if progress else None
        # the commit follow-up and the v2 summary are extra (short) model calls;
        # count and charge them
        D.on_followup = self._count_followup
        D.on_summary = self._count_summary

    def _postfix(self) -> None:
        """One place for the bar's extra fields, so setting one never drops another."""
        if self._bar is None:
            return
        extra = {}
        if self.errors:
            extra["err"] = self.errors
        if self.followups:
            extra["followups"] = self.followups
        if self.summaries:
            extra["summaries"] = self.summaries
        self._bar.set_postfix(**extra, refresh=False)

    def _count_followup(self) -> None:
        with self._lock:
            self.followups += 1
        self.charge(1)
        self._postfix()

    def _count_summary(self) -> None:
        with self._lock:
            self.summaries += 1
        self.charge(1)
        self._postfix()

    def stage(self, label: str) -> None:
        if self._bar is not None:
            self._bar.set_description(label[:48])

    def _client(self):
        with self._lock:
            return next(self._clients)

    def charge(self, cost: int) -> None:
        """Charge `cost` calls against the budget (and the bar). Public so callers
        can account for extra calls of their own, e.g. treegrow's controller."""
        with self._lock:
            if self.max_calls is not None and self.calls + cost > self.max_calls:
                raise BudgetExhausted(f"call budget {self.max_calls} exhausted at {self.calls}")
            self.calls += cost
            if self._bar is not None:
                self._bar.update(cost)

    def lookup(self, qid: str, executed_specs: list[dict], round_spec: dict, rep: int = 0,
               prompts: dict | None = None) -> tuple[bool, list[str], list | None, list[int]]:
        """(per_speaker, keys, the cached (persona, response) pairs or None,
        indices of the speakers not yet recorded). A blind round under v3 is
        recorded speaker by speaker, so some of its speakers may be recorded and
        others not; any other round is one recording."""
        per_speaker, keys = round_keys(executed_specs, round_spec, prompts)
        with self._lock:
            hits = [self._cache.get((qid, k, rep)) for k in keys]
        if not per_speaker:
            if hits[0] is None:
                return False, keys, None, list(range(len(round_spec["personas"])))
            return False, keys, [tuple(pr) for pr in hits[0]], []
        missing = [i for i, h in enumerate(hits) if h is None]
        return True, keys, (None if missing else [tuple(h[0]) for h in hits]), missing

    def round_tokens(self, qid: str, executed_specs: list[dict], round_spec: dict, rep: int = 0,
                     prompts: dict | None = None) -> int:
        """Completion tokens the recording(s) of this round used (0 if not logged)."""
        _, keys = round_keys(executed_specs, round_spec, prompts)
        with self._lock:
            return sum(self._tokens.get((qid, k, rep), 0) for k in keys)

    def round_summary_tokens(self, qid: str, executed_specs: list[dict], round_spec: dict, rep: int = 0,
                             prompts: dict | None = None) -> int:
        """Of round_tokens, the tokens of the round's answer-locked summaries (0 if not recorded apart)."""
        _, keys = round_keys(executed_specs, round_spec, prompts)
        with self._lock:
            return sum(self._summary_tokens.get((qid, k, rep), 0) for k in keys)

    def run_round(self, qid: str, all_rounds: list, executed_specs: list[dict],
                  round_spec: dict, rep: int = 0,
                  prompts: dict | None = None) -> list[tuple[str, str]]:
        """Execute `round_spec` for `qid` on top of `all_rounds` (the transcript
        of `executed_specs`). Cached prefixes are free. Returns the new round's
        (persona, response) pairs; an errored round returns []. Of a blind round
        (v3), only the speakers not yet recorded are run, each recorded alone.
        If another thread recorded the same key first, its recording is kept and
        returned, so what a debate saw is always what a later replay reads."""
        per_speaker, keys, hit, missing = self.lookup(qid, executed_specs, round_spec, rep, prompts)
        if hit is not None:
            return hit
        spec = round_spec
        if per_speaker:
            spec = dict(round_spec, personas=[round_spec["personas"][i] for i in missing])
        row = self.rows[qid]
        self.charge(len(spec["personas"]))
        err, responses = None, []
        for _ in range(2):                                # one retry on a transport hiccup
            try:
                responses = D.execute_round(self._client(), self.model, row["question"],
                                            list(row["options"]), spec, all_rounds,
                                            self.temperature, self.answer_tokens, prompts,
                                            prior_specs=executed_specs)
                err = None
                break
            except Exception as exc:
                err = str(exc)
        usage = None if err else D.last_round_usage()
        if per_speaker:
            recs = [{"q": qid, "k": keys[i], "r": rep,
                     "responses": [] if err else [list(responses[j])], "error": err}
                    for j, i in enumerate(missing)]
            for j, rec in enumerate(recs):
                if usage:
                    rec["usage"] = [usage[j]]
        else:
            recs = [{"q": qid, "k": keys[0], "r": rep,
                     "responses": [list(pr) for pr in responses], "error": err}]
            if usage:
                recs[0]["usage"] = usage          # tokens per speaker; for reporting only
        with self._lock:
            self.rounds_run += 1
            if err:
                self.errors += 1
            for rec in recs:
                key = (qid, rec["k"], rep)
                if key in self._cache:            # another thread recorded it first
                    continue
                if not err:
                    self._cache[key] = rec["responses"]
                    if "usage" in rec:
                        self._tokens[key] = sum((u or {}).get("completion", 0) for u in rec["usage"])
                        self._summary_tokens[key] = sum((u or {}).get("summary", 0) for u in rec["usage"])
                self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            if not err:
                if per_speaker:
                    out = [tuple(self._cache[(qid, k, rep)][0]) for k in keys]
                else:
                    out = [tuple(pr) for pr in self._cache[(qid, keys[0], rep)]]
        self._postfix()
        return [] if err else out

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        if not self._fh.closed:
            self._fh.close()


# --- structural variation operators (no LLM) -------------------------------
# Deliberately LLM-free: if random structural variation illuminates the space as
# well as the gpt-5.4-mini architect did, that is itself a result about how much
# the "guided" in "guided structural search" was worth.

def _rand_personas(rng: random.Random, personas=None, weights=None) -> list[str]:
    n = rng.choices([1, 2, 3, 4], weights=[0.55, 0.25, 0.13, 0.07])[0]
    pool = list(personas or D.EXT_PERSONAS)
    return [rng.choices(pool, weights=weights)[0] for _ in range(n)]


def _rand_round(rng: random.Random) -> dict:
    r = {"personas": _rand_personas(rng)}
    sees = rng.choices(list(D.SEES), weights=[0.5, 0.15, 0.12, 0.13, 0.10])[0]
    if sees != D.DEFAULT_SEES:
        r["sees"] = sees
    return r


def repair(schema: dict) -> dict | None:
    """Make a candidate grammar-valid if a cheap fix exists, else None."""
    rounds = []
    for r in schema["rounds"][: D.MAX_ROUNDS]:
        if not r.get("personas"):
            continue
        nr = {"personas": list(r["personas"][: D.MAX_PERSONAS])}
        if r.get("sees", D.DEFAULT_SEES) in D.SEES and r.get("sees", D.DEFAULT_SEES) != D.DEFAULT_SEES:
            nr["sees"] = r["sees"]
        rounds.append(nr)
    s = {"rounds": rounds, "final": schema["final"] if schema["final"] in D.FINALS else "last"}
    if not s["rounds"]:
        return None
    last = s["rounds"][-1]["personas"]
    if all(p in D.NON_ANSWERING for p in last):        # the last round must commit something
        if len(last) < D.MAX_PERSONAS:
            last.append("solver")
        else:
            last[-1] = "solver"
    if s["final"] == "synthesizer" and "synthesizer" not in last:
        if len(last) < D.MAX_PERSONAS:
            last.append("synthesizer")
        else:
            s["final"] = "last"
    return s if D.validate(s) else None


def mutate(schema: dict, rng: random.Random) -> dict | None:
    s = json.loads(json.dumps(schema))
    rounds = s["rounds"]
    ops = ["add_round", "modify_round", "set_final", "add_persona", "remove_persona",
           "set_sees", "set_sees"]                     # the new gene, weighted up while it is novel
    if len(rounds) > 1:
        ops += ["remove_round", "remove_round"]        # shrink is under-used; weight it up
    op = rng.choice(ops)
    if op == "add_round" and len(rounds) < D.MAX_ROUNDS:
        rounds.insert(rng.randint(0, len(rounds)), _rand_round(rng))
    elif op == "modify_round":
        rounds[rng.randrange(len(rounds))] = _rand_round(rng)
    elif op == "remove_round" and len(rounds) > 1:
        rounds.pop(rng.randrange(len(rounds)))
    elif op == "set_final":
        s["final"] = rng.choice(list(D.FINALS))
    elif op == "set_sees":
        r = rounds[rng.randrange(len(rounds))]
        sees = rng.choice(list(D.SEES))
        r.pop("sees", None)
        if sees != D.DEFAULT_SEES:
            r["sees"] = sees
    elif op == "add_persona":
        r = rounds[rng.randrange(len(rounds))]
        if len(r["personas"]) < D.MAX_PERSONAS:
            r["personas"].insert(rng.randint(0, len(r["personas"])),
                                 rng.choice(list(D.EXT_PERSONAS)))
    elif op == "remove_persona":
        r = rounds[rng.randrange(len(rounds))]
        if len(r["personas"]) > 1:
            r["personas"].pop(rng.randrange(len(r["personas"])))
    return repair(s)


def crossover(a: dict, b: dict, rng: random.Random) -> dict | None:
    """Single-point round splice -- the recombination the current loop has none of."""
    ra, rb = a["rounds"], b["rounds"]
    i, j = rng.randint(1, len(ra)), rng.randint(0, len(rb) - 1)
    child = {"rounds": json.loads(json.dumps(ra[:i] + rb[j:]))[: D.MAX_ROUNDS],
             "final": rng.choice([a["final"], b["final"]])}
    return repair(child)

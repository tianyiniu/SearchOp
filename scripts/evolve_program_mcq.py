"""Evolve debate-control programs, scored by replaying cached transcripts.

A program is an ordered list of if-then rules. After every executed round it is
consulted with the observable state of the debate (round count, committed
letters, what ran last) and answers with one action: run the next planned
round, run a named extra round, or stop and read off the answer. One program
serves every question, but each question takes its own path through the rules.
No answer key is used at decision time.

Scoring is by REPLAY: the round caches written by the adaptive and treegrow
runs store every executed round keyed by (question, exact round sequence,
replicate). Whatever round a program asks for, we look up the recorded result
instead of calling the model. A program whose path was never recorded is
counted as off-cache, not silently wrong.

LIVE MODE (--live) removes that limit. Off-cache rounds are executed by the
model, written to a NEW cache file, and reused by every later program. Three
guards keep it honest and affordable:

  prescreen      a candidate is first scored on the existing cache for free;
                 it may spend live calls only if the score it could reach
                 after the fills its budget affords is within
                 --prescreen-margin of the population cutoff, and it covers
                 at least --min-coverage of the batch. Junk mutants that go
                 off-cache on every question never pay.
  novelty budget at most --novelty-budget new calls per candidate. The
                 budget decides how many off-cache questions get admitted
                 (at the program's usual cost per question); those run to
                 completion, the rest stay off-cache and count as wrong.
  recheck        the final top programs are re-run on the dev half with a
                 fresh replicate (every round re-sampled, nothing reused), so a
                 program that only looked good because of one lucky new round
                 is caught before its number is reported.
A per-question call cap (--max-calls-per-question) stops runaway programs.

    python3 scripts/evolve_program_mcq.py --validate
    python3 scripts/evolve_program_mcq.py --evolve --save-all outputs/x.jsonl
    python3 scripts/evolve_program_mcq.py --evolve --live ...   # needs vLLM

Validation encodes experiment B's hand-written rules (adaptive_debate_mcq.py)
as PROGRAM_B and replays them over B's own round cache. It must reproduce B's
recorded per-question results exactly (letter, track, closer), and the fixed
master recipe must reproduce the recorded 14.5%. Only after this passes is the
replay trustworthy enough to put a search on top of it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D
import schema_fitness as SF
import adaptive_debate_mcq as B

MAX_STEPS = 12                     # safety cap on rounds per question


# --- actions ----------------------------------------------------------------
# "continue" runs the next round of the program's plan. Every other action runs
# one or more fixed extra rounds. The menu is the union of what experiment B
# used and what treegrow's moves used, and the specs are copied byte-for-byte
# from those scripts: the cache keys hash the specs, so any deviation makes a
# recorded round unfindable.

ACTIONS: dict[str, list[dict]] = {
    # treegrow's moves (treegrow_mcq.ACTIONS)
    "critic":      [{"personas": ["critic"], "sees": "all"}],
    "verifier":    [{"personas": ["verifier"], "sees": "all"}],
    "fresh":       [{"personas": ["independent", "independent"]}],   # forced blind
    "expert_blind": [{"personas": ["expert"], "sees": "none"}],
    "eliminate":   [{"personas": ["eliminator"], "sees": "all"},
                    {"personas": ["solver"], "sees": "all"}],
    # experiment B's extension rounds (adaptive_debate_mcq.run_question)
    "elim_blind":  [{"personas": ["eliminator"], "sees": "none"}],
    "pair_expert_solver": [{"personas": ["expert", "solver"], "sees": "last_round"}],
    "synthesizer": [{"personas": ["synthesizer"], "sees": "all"}],
}

# Moves built on the eliminator. It refuses to answer and emits a survivor
# list instead, which is awkward to justify in a deployed system, and the
# measurements are against it too: worst verified solve rate of any move
# (0.3%), 81% of the time it struck out the correct option, and its survivor
# list reached the next persona only 2% of the time under the default digest
# window. --no-eliminator drops these moves and swaps in eliminator-free seeds.
ELIMINATOR_ACTIONS = ("eliminate", "elim_blind")

# How a stop reads the answer off the transcript:
#   last_commit   the most recent committed letter
#   last_speaker  the last round's letter under debate_mcq.final_letter("last")
#   vote          plurality over EVERY committed letter in the transcript (ties go
#                 to the letter committed first). On the 27B dev transcripts this
#                 beat last_commit by 1.3 points at no extra cost: the personas
#                 flip answers at chance precision, so the last word is no better
#                 than any other word, and counting all of them averages the noise.
STOP_READS = ("last_commit", "last_speaker", "vote")


# --- observable state and conditions ----------------------------------------

class State:
    """Everything a rule may look at. Built fresh after each round."""

    def __init__(self, rounds: list, specs: list, actions: list[str],
                 plan: list[dict], n: int):
        self.rounds, self.specs, self.actions = rounds, specs, actions
        self.plan, self.n = plan, n
        self.step = len(rounds)
        self.plan_pos = actions.count("continue")
        self.extended = any(a != "continue" for a in actions)
        self.commits = B.committed_with_round(rounds, n)
        r1 = [l for ri, l in self.commits if ri == 0]
        self.baseline = B.modal_letter(r1) if self.step >= 1 else None
        # round-1 majority size (4 = unanimous ... 1 = all different; 0 = not run)
        self.r1_majority = max(Counter(r1).values()) if r1 else 0
        self.n_distinct = len({l for _, l in self.commits})
        # what each executed round WAS, by persona content ("solver", "critic",
        # "expert+solver", ...). Plan rounds are all the action "continue", so
        # without this a rule could not ask "was the last round a critic".
        self.round_kinds = [round_kind(s) for s in specs]


def round_kind(spec: dict) -> str:
    return "+".join(dict.fromkeys(spec["personas"]))


ROUND_KINDS = sorted({round_kind(s) for specs in ACTIONS.values() for s in specs}
                     | {round_kind(s) for s in B.MASTER_ROUNDS})

_NUMERIC = {"step": lambda st: st.step,
            "acts": lambda st: len(st.actions),
            "r1_majority": lambda st: st.r1_majority,
            "n_distinct": lambda st: st.n_distinct}
_NUM_RE = re.compile(r"(step|acts|r1_majority|n_distinct)(==|>=|<=|<|>)(\d+)")
_BOOL_CONDS = ("plan_left", "not_extended", "confirmed_switch", "parked",
               "last_round_agree", "verifier_backed")


def check_condition(cond: str) -> None:
    """Reject a condition the interpreter would not understand, at validation
    time rather than in the middle of a run."""
    if _NUM_RE.fullmatch(cond) or cond in _BOOL_CONDS:
        return
    head, _, tail = cond.partition(":")
    if head in ("last", "ran") and (tail == "continue" or tail in ACTIONS):
        return
    if head in ("last_round", "ran_round") and tail in ROUND_KINDS:
        return
    raise ValueError(f"unknown condition {cond!r}")


def _cond(cond: str, st: State) -> bool:
    if m := _NUM_RE.fullmatch(cond):
        v = _NUMERIC[m.group(1)](st)
        op, k = m.group(2), int(m.group(3))
        return {"==": v == k, ">=": v >= k, "<=": v <= k,
                "<": v < k, ">": v > k}[op]
    if cond == "plan_left":
        return not st.extended and st.plan_pos < len(st.plan)
    if cond == "not_extended":
        return not st.extended
    if cond == "confirmed_switch":
        return B.is_confirmed_switch(st.commits, st.baseline)
    if cond == "parked":                       # no post-round-1 commit left the baseline
        post = [l for ri, l in st.commits if ri >= 1]
        return not post or all(l == st.baseline for l in post)
    if cond.startswith("last:"):
        return bool(st.actions) and st.actions[-1] == cond[5:]
    if cond.startswith("ran:"):
        return cond[4:] in st.actions
    if cond.startswith("last_round:"):          # by round content, plan rounds included
        return bool(st.round_kinds) and st.round_kinds[-1] == cond[11:]
    if cond.startswith("ran_round:"):
        return cond[10:] in st.round_kinds
    if cond == "last_round_agree":             # last round: >=2 speakers, all parseable, all equal
        if not st.rounds:
            return False
        letters = [D.extract_letter(r, st.n) for p, r in st.rounds[-1]
                   if p not in D.NON_ANSWERING]
        return len(letters) >= 2 and None not in letters and len(set(letters)) == 1
    if cond == "verifier_backed":              # last round's letter repeats an earlier commit
        if not st.rounds:
            return False
        prior = {l for ri, l in st.commits if ri < st.step - 1}
        last = [D.extract_letter(r, st.n) for p, r in st.rounds[-1]
                if p not in D.NON_ANSWERING]
        return len(last) == 1 and last[0] is not None and last[0] in prior
    raise ValueError(f"unknown condition {cond!r}")


def validate_program(prog: dict) -> None:
    """Raise (AssertionError or ValueError) on anything the interpreter could
    not run: an unknown action, an unknown condition, or a default that is not
    a stop (the default is the only guaranteed way out of the loop)."""
    default = prog["default"]
    assert default.startswith("stop:") and default[5:] in STOP_READS, default
    assert prog.get("plan"), "a program needs a plan"
    for rule in prog["rules"]:
        act = rule["do"]
        if act.startswith("stop:"):
            assert act[5:] in STOP_READS, act
        else:
            assert act == "continue" or act in ACTIONS, act
        for cond in rule["when"]:
            check_condition(cond)


# --- the replay engine ------------------------------------------------------

class OffCache(Exception):
    pass


def well_formed(rec: dict) -> bool:
    """A cache record two processes tore can still parse as JSON with a broken
    body. Accept only what the runner actually writes: a question id, a key
    that is itself JSON, a replicate number, and responses as (persona, text)
    pairs with a non-empty persona."""
    try:
        if not (isinstance(rec.get("q"), str) and isinstance(rec.get("k"), str)
                and isinstance(rec.get("r"), int) and isinstance(rec.get("responses"), list)):
            return False
        json.loads(rec["k"])
        return all(isinstance(pr, (list, tuple)) and len(pr) == 2 and isinstance(pr[0], str)
                   and pr[0] and isinstance(pr[1], str) for pr in rec["responses"])
    except (ValueError, TypeError):
        return False


def load_cache_into(target: dict, path: Path) -> None:
    """Read one recorded-rounds file into `target` (first write wins)."""
    bad = 0
    for line in tqdm(Path(path).open(errors="replace"),
                     desc=f"load {Path(path).name}", unit="line",
                     miniters=100_000, leave=False):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if not isinstance(r, dict) or not well_formed(r):
            bad += 1
            continue
        if not r.get("error"):
            target.setdefault((r["q"], r["k"], r["r"]), r["responses"])
    print(f"loaded {len(target)} rounds so far from {path}"
          + (f" ({bad} unreadable or malformed lines skipped)" if bad else ""))


class CacheRunner:
    """Read-only stand-in for schema_fitness.RoundRunner: run_round is a pure
    lookup into the recorded round caches. A missing key raises OffCache."""

    def __init__(self, cache_paths: list[Path]):
        self._cache: dict[tuple, list] = {}
        for path in cache_paths:
            load_cache_into(self._cache, path)

    def run_round(self, qid: str, all_rounds: list, executed_specs: list[dict],
                  round_spec: dict, rep: int = 0, prompts: dict | None = None) -> list:
        key = (qid, SF.path_key(executed_specs + [round_spec], prompts), rep)
        hit = self._cache.get(key)
        if hit is None:
            raise OffCache(key)
        return [tuple(pr) for pr in hit]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                       # exists, owned by someone else


class BudgetedRunner(SF.RoundRunner):
    """The real round runner (cache first, model on a miss) with a novelty
    budget. Reads the recorded caches of earlier experiments read-only; writes
    only to its own cache file, so those recordings are never touched.

    `novelty_budget` is the number of NEW model calls this runner may make
    before a miss goes back to raising OffCache. Set it to 0 and the runner
    behaves exactly like CacheRunner; set it to None for no limit (the recheck
    pass). The counter is shared across threads."""

    def __init__(self, rows, read_only_caches: list[Path], lock: bool = True, **kw):
        # Two processes appending to one cache file tear its lines (seen when
        # two servers shared one path over a network volume). A lock FILE is
        # the guard that works on shared volumes; OS file locks often do not.
        self._lock_path = Path(kw["cache_path"]).with_suffix(".lock")
        self._own_lock = False
        if lock:
            self._acquire_lock(kw["cache_path"])
        try:
            super().__init__(rows, **kw)
        except BaseException:
            self._release_lock()          # a failed or interrupted start leaves no lock
            raise
        # the base runner loaded this runner's own cache file without shape
        # checks; drop anything a torn write left behind
        broken = [k for k, v in self._cache.items()
                  if not well_formed({"q": k[0], "k": k[1], "r": k[2], "responses": v})]
        for k in broken:
            del self._cache[k]
        if broken:
            print(f"dropped {len(broken)} malformed rounds from {kw['cache_path']}")
        for path in read_only_caches:
            load_cache_into(self._cache, path)
        self.novelty_budget: int | None = 0
        self.novel_calls = 0
        self._novel_lock = threading.Lock()

    def close(self) -> None:
        super().close()
        self._release_lock()

    def _acquire_lock(self, cache_path) -> None:
        for attempt in range(2):
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:
                    holder = json.loads(self._lock_path.read_text())
                except (OSError, ValueError):
                    holder = {}
                same_host = holder.get("host") == socket.gethostname()
                alive = same_host and isinstance(holder.get("pid"), int) and _pid_alive(holder["pid"])
                if same_host and not alive and attempt == 0:
                    print(f"clearing stale lock left by dead pid {holder.get('pid')} "
                          f"(started {holder.get('started')})")
                    self._lock_path.unlink(missing_ok=True)
                    continue
                raise SystemExit(
                    f"live cache {cache_path} is in use by another run: {holder or 'unknown'}\n"
                    f"Two runs writing one cache tear its lines. Give this run its own "
                    f"--live-cache, or pass --ignore-cache-lock if that run is dead."
                    + ("" if same_host else " (it is on another host, so liveness cannot be checked here)"))
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                                 "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
        self._own_lock = True

    def _release_lock(self) -> None:
        if self._own_lock:
            self._lock_path.unlink(missing_ok=True)
            self._own_lock = False

    def reset_budget(self, budget: int | None) -> None:
        with self._novel_lock:
            self.novelty_budget, self.novel_calls = budget, 0

    def run_round(self, qid, all_rounds, executed_specs, round_spec, rep=0, prompts=None):
        key = (qid, SF.path_key(executed_specs + [round_spec], prompts), rep)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return [tuple(pr) for pr in hit]
        cost = len(round_spec["personas"])
        with self._novel_lock:
            if self.novelty_budget is not None and self.novel_calls + cost > self.novelty_budget:
                raise OffCache(key)
            self.novel_calls += cost
        out = super().run_round(qid, all_rounds, executed_specs, round_spec,
                                rep=rep, prompts=prompts)
        if not out:                          # errored round: not cached, not usable
            raise OffCache(key)
        return out


def run_program(prog: dict, runner, row: dict, rep: int = 0,
                max_calls: int | None = None) -> dict:
    """Drive one question with a program. `runner` is a CacheRunner (replay) or
    a live runner (both expose run_round with the same signature). With
    `max_calls`, an action that would push the question past that many model
    calls is replaced by the program's default stop."""
    qid, n = row["id"], len(row["options"])
    prompts = B.question_prompts(row)
    plan = prog["plan"]
    rounds: list = []
    specs: list[dict] = []
    actions: list[str] = []
    read = prog["default"].removeprefix("stop:")

    for _ in range(MAX_STEPS):
        st = State(rounds, specs, actions, plan, n)
        act = prog["default"]
        for rule in prog["rules"]:
            if all(_cond(c, st) for c in rule["when"]):
                act = rule["do"]
                break
        if act == "continue" and st.plan_pos >= len(plan):
            act = prog["default"]              # nothing left to continue: stop
        if not act.startswith("stop:") and max_calls is not None:
            todo = [plan[st.plan_pos]] if act == "continue" else ACTIONS[act]
            spent = sum(D.turn_cost(s["personas"]) for s in specs)
            if spent + sum(D.turn_cost(s["personas"]) for s in todo) > max_calls:
                act = prog["default"]          # over the per-question cap: stop
        if act.startswith("stop:"):
            read = act[5:]
            break
        for spec in [plan[st.plan_pos]] if act == "continue" else ACTIONS[act]:
            rounds.append(runner.run_round(qid, rounds, specs, spec, rep=rep,
                                           prompts=prompts))
            specs.append(spec)
        actions.append(act)

    if read == "last_speaker":
        final = D.final_letter("last", rounds, n) if rounds else None
    elif read == "vote":
        letters = SF.committed_letters(rounds, n)
        final = Counter(letters).most_common(1)[0][0] if letters else None
    else:                                                  # "last_commit"
        letters = SF.committed_letters(rounds, n)
        final = letters[-1] if letters else None
    gold = row.get("answer_letter")
    return {"qid": qid, "letter": final,
            "correct": final is not None and final == gold,
            "actions": actions,
            "n_calls": sum(D.turn_cost(s["personas"]) for s in specs)}


# --- experiment B and the fixed recipe, expressed as programs ---------------

PROGRAM_B = {
    "plan": B.MASTER_ROUNDS,
    "rules": [
        # master recipe: run round 1, then continue until a switch is confirmed
        {"when": ["step<=1"], "do": "continue"},
        {"when": ["not_extended", "confirmed_switch"], "do": "stop:last_commit"},
        {"when": ["plan_left"], "do": "continue"},
        # parked extension: eliminator -> expert+solver pair -> verifier
        {"when": ["not_extended", "parked"], "do": "elim_blind"},
        {"when": ["last:elim_blind"], "do": "pair_expert_solver"},
        {"when": ["last:pair_expert_solver", "last_round_agree"], "do": "stop:last_commit"},
        {"when": ["last:pair_expert_solver"], "do": "verifier"},
        {"when": ["ran:pair_expert_solver", "last:verifier"], "do": "stop:last_commit"},
        # churn extension: verifier -> synthesizer
        {"when": ["not_extended"], "do": "verifier"},
        {"when": ["last:verifier", "verifier_backed"], "do": "stop:last_commit"},
        {"when": ["last:verifier"], "do": "synthesizer"},
    ],
    "default": "stop:last_commit",
}

PROGRAM_FIXED = {
    "plan": B.MASTER_ROUNDS,
    "rules": [{"when": ["plan_left"], "do": "continue"}],
    "default": "stop:last_speaker",
}

# Experiment B's shape without the eliminator. The settled and churn tracks are
# unchanged; the parked track goes straight to the fresh outside voices that
# the live search found on its own (blind expert, then the expert+solver pair),
# instead of routing them through an eliminator whose survivor list they could
# not see anyway.
PROGRAM_B_NOELIM = {
    "plan": B.MASTER_ROUNDS,
    "rules": [
        {"when": ["step<=1"], "do": "continue"},
        {"when": ["not_extended", "confirmed_switch"], "do": "stop:last_commit"},
        {"when": ["plan_left"], "do": "continue"},
        # parked: bring in voices that have not seen the stuck answer
        {"when": ["not_extended", "parked"], "do": "expert_blind"},
        {"when": ["last:expert_blind"], "do": "pair_expert_solver"},
        {"when": ["last:pair_expert_solver", "last_round_agree"], "do": "stop:last_commit"},
        {"when": ["last:pair_expert_solver"], "do": "verifier"},
        {"when": ["ran:pair_expert_solver", "last:verifier"], "do": "stop:last_commit"},
        # churn: judge between the candidates already on the table
        {"when": ["not_extended"], "do": "verifier"},
        {"when": ["last:verifier", "verifier_backed"], "do": "stop:last_commit"},
        {"when": ["last:verifier"], "do": "synthesizer"},
    ],
    "default": "stop:last_commit",
}


def with_vote_read(prog: dict) -> dict:
    """The same program with every stop reading the plurality of all commits."""
    p = json.loads(json.dumps(prog))
    for rule in p["rules"]:
        if rule["do"].startswith("stop:"):
            rule["do"] = "stop:vote"
    p["default"] = "stop:vote"
    return p


PROGRAM_B_VOTE = with_vote_read(PROGRAM_B)
PROGRAM_B_NOELIM_VOTE = with_vote_read(PROGRAM_B_NOELIM)
PROGRAM_FIXED_VOTE = with_vote_read(PROGRAM_FIXED)


def drop_eliminator() -> None:
    """Remove the eliminator moves from the action menu for this process. Any
    seed or mutation that used them is rebuilt without them."""
    for name in ELIMINATOR_ACTIONS:
        ACTIONS.pop(name, None)
    global ROUND_KINDS
    ROUND_KINDS = sorted({round_kind(s) for specs in ACTIONS.values() for s in specs}
                         | {round_kind(s) for s in B.MASTER_ROUNDS})


def add_deep_think() -> None:
    """Add the deep-think move to the action menu for this process (see
    debate_mcq: one speaker with the model's highest reasoning setting and no
    reply cap; one such turn counts as debate_mcq.DEEP_COST turns). Use
    program_space.configure_executor, which also switches the persona on."""
    ACTIONS[D.DEEP_PERSONA] = [{"personas": [D.DEEP_PERSONA], "sees": "all"}]
    global ROUND_KINDS
    ROUND_KINDS = sorted({round_kind(s) for specs in ACTIONS.values() for s in specs}
                         | {round_kind(s) for s in B.MASTER_ROUNDS})


def derive_track(actions: list[str], plan_len: int) -> tuple[str, str | None]:
    """Reconstruct experiment B's track/closer labels from a replayed path, for
    comparison against its records file."""
    n_plan = sum(1 for a in actions if a == "continue")
    if "elim_blind" in actions:
        closer = "newcomers_agree" if actions[-1] == "pair_expert_solver" else "verifier"
        return "parked", closer
    if "verifier" in actions:
        closer = "verifier_backed" if actions[-1] == "verifier" else "synthesizer"
        return "churn", closer
    return ("settled_full" if n_plan == plan_len else "settled_early"), None


# --- coverage probes over the treegrow cache --------------------------------

TG_ROOT = {"personas": ["solver"]}                 # treegrow_mcq.ROOT_SPEC


def chain(*acts: str) -> dict:
    """A straight-line program: root solver, then the given actions, then stop."""
    return {"plan": [TG_ROOT],
            "rules": [{"when": [f"acts=={i}"], "do": a}
                      for i, a in enumerate(("continue",) + acts)],
            "default": "stop:last_commit"}


PROBES: dict[str, dict] = {
    "solver_critic":          chain("critic"),
    "solver_critic2":         chain("critic", "critic"),
    "solver_verifier_critic": chain("verifier", "critic"),
    "solver_expert":          chain("expert_blind"),
    "solver_critic4":         chain("critic", "critic", "critic", "critic"),
    "branch_on_parked": {
        "plan": [TG_ROOT],
        "rules": [{"when": ["acts==0"], "do": "continue"},
                  {"when": ["acts==1"], "do": "critic"},
                  {"when": ["acts==2", "parked"], "do": "fresh"},
                  {"when": ["acts==2"], "do": "verifier"}],
        "default": "stop:last_commit"},
}


def coverage(args) -> None:
    """Replay the probe programs against the treegrow cache. Reports how much
    of each program's path is covered by recorded rounds, and its accuracy.
    Off-cache questions are counted as wrong in the headline number."""
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    runner = CacheRunner([args.treegrow_cache])
    print(f"\n{'program':24}{'covered':>10}{'acc(all)':>10}{'acc(covered)':>14}{'calls':>7}")
    for name, prog in PROBES.items():
        validate_program(prog)
        done = correct = calls = 0
        for qid, row in tqdm(rows.items(), desc=name, unit="q", leave=False):
            try:
                out = run_program(prog, runner, row, rep=args.rep)
            except OffCache:
                continue
            done += 1
            correct += out["correct"]
            calls += out["n_calls"]
        n = len(rows)
        print(f"{name:24}{done / n:>10.1%}{correct / n:>10.1%}"
              f"{(correct / done if done else 0):>14.1%}"
              f"{(calls / done if done else 0):>7.1f}")


# --- evolution --------------------------------------------------------------
# Random structural mutations over the rule language, scored by replay. No LLM
# in the loop for now: if random edits already find programs that beat the
# hand-written ones, an informed mutator can only help; if the space is flat,
# better to learn that at zero cost. Selection is on the train half only; the
# dev half is reported but never selected on.

MAX_RULES = 14


def random_condition(rng: random.Random) -> str:
    kind = rng.randrange(8)
    if kind == 0:
        return f"acts{rng.choice(['==', '>='])}{rng.randrange(6)}"
    if kind == 1:
        return f"step{rng.choice(['==', '>=', '<'])}{rng.randrange(7)}"
    if kind == 6:                      # behavior signals observable at run time
        if rng.random() < 0.5:
            return f"r1_majority{rng.choice(['==', '>=', '<'])}{rng.randrange(1, 5)}"
        return f"n_distinct{rng.choice(['==', '>='])}{rng.randrange(1, 5)}"
    if kind == 7:                      # by round content, plan rounds included
        return f"{rng.choice(['last_round', 'ran_round'])}:{rng.choice(ROUND_KINDS)}"
    if kind == 2:
        return rng.choice(["parked", "not_extended", "confirmed_switch",
                           "last_round_agree", "verifier_backed", "plan_left"])
    if kind == 3:
        return f"last:{rng.choice(list(ACTIONS))}"
    if kind == 4:
        return f"ran:{rng.choice(list(ACTIONS))}"
    return f"acts=={rng.randrange(6)}"


def random_action(rng: random.Random) -> str:
    acts = ["continue", "stop:last_commit", "stop:last_speaker", "stop:vote"] + list(ACTIONS)
    return rng.choice(acts)


def mutate_program(prog: dict, rng: random.Random, rand_cond=None, rand_act=None,
                   defaults: tuple[str, ...] = ("stop:last_commit", "stop:last_speaker",
                                                "stop:vote"),
                   ) -> dict:
    """One random edit. The condition/action generators are parameters so the
    FRAMES search programs (evolve_program_frames.py) can reuse this with their
    own menus."""
    rand_cond = rand_cond or random_condition
    rand_act = rand_act or random_action
    p = json.loads(json.dumps(prog))               # deep copy
    rules = p["rules"]
    ops = ["change_action", "add_rule", "change_cond", "change_default"]
    if len(rules) > 1:
        ops += ["drop_rule", "swap_rules"]
    op = rng.choice(ops)
    if op == "change_action" and rules:
        rng.choice(rules)["do"] = rand_act(rng)
    elif op == "add_rule" and len(rules) < MAX_RULES:
        rule = {"when": [rand_cond(rng) for _ in range(rng.choice([1, 1, 2]))],
                "do": rand_act(rng)}
        rules.insert(rng.randrange(len(rules) + 1), rule)
    elif op == "change_cond" and rules:
        rule = rng.choice(rules)
        if rule["when"] and rng.random() < 0.7:
            rule["when"][rng.randrange(len(rule["when"]))] = rand_cond(rng)
        else:
            rule["when"] = rule["when"] + [rand_cond(rng)]
    elif op == "change_default":
        p["default"] = rng.choice(list(defaults))
    elif op == "drop_rule":
        rules.pop(rng.randrange(len(rules)))
    elif op == "swap_rules":
        i, j = rng.sample(range(len(rules)), 2)
        rules[i], rules[j] = rules[j], rules[i]
    return p


def canon_prog(prog: dict) -> str:
    return json.dumps(prog, sort_keys=True, separators=(",", ":"))


class Eval:
    """Per-question record of one program over one list of questions, kept so
    routing and clustering analyses can use every scored program without
    replaying it. Aligned to `qids`: marks ('1' correct, '0' wrong, 'x'
    off-cache), the final letter ('?' if none), calls spent, and the action
    path (as an index into `paths`)."""

    def __init__(self, n: int):
        self.marks = ["x"] * n
        self.letters = ["?"] * n
        self.calls = [0] * n
        self.path_ids = [-1] * n
        self.paths: list[str] = []
        self._path_index: dict[str, int] = {}

    def record(self, i: int, out: dict) -> None:
        self.marks[i] = "1" if out["correct"] else "0"
        self.letters[i] = out["letter"] or "?"
        self.calls[i] = out["n_calls"]
        path = ",".join(out["actions"])
        if path not in self._path_index:
            self._path_index[path] = len(self.paths)
            self.paths.append(path)
        self.path_ids[i] = self._path_index[path]

    @property
    def correct(self) -> int:
        return self.marks.count("1")

    @property
    def covered(self) -> int:
        return len(self.marks) - self.marks.count("x")

    def to_json(self, prefix: str) -> dict:
        return {f"{prefix}_n_correct": self.correct, f"{prefix}_n_covered": self.covered,
                f"{prefix}_outcomes": "".join(self.marks),
                f"{prefix}_letters": "".join(self.letters),
                f"{prefix}_calls": self.calls, f"{prefix}_path_ids": self.path_ids,
                f"{prefix}_paths": self.paths}


def eval_program(prog: dict, runner, rows: dict, qids: list[str], rep: int,
                 max_calls: int | None = None, workers: int = 1,
                 only: list[int] | None = None, into: Eval | None = None) -> Eval:
    """Score `prog` on `qids`. Off-cache counts as wrong. `only` restricts to
    those indices (the live fill re-runs just the off-cache questions);
    `into` merges results into an existing Eval. `workers` > 1 runs questions
    in parallel, which only makes sense when the runner may call the model."""
    ev = into or Eval(len(qids))
    idx = list(range(len(qids))) if only is None else only

    def one(i: int):
        try:
            return i, run_program(prog, runner, rows[qids[i]], rep=rep, max_calls=max_calls)
        except OffCache:
            return i, None

    if workers > 1 and len(idx) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(pool.map(one, idx), total=len(idx), unit="q",
                                desc="questions", leave=False, disable=len(idx) < 50))
    else:
        results = [one(i) for i in tqdm(idx, unit="q", desc="questions", leave=False,
                                        disable=len(idx) < 50)]
    for i, out in results:
        if out is not None:
            ev.record(i, out)
        elif into is None:
            ev.marks[i] = "x"
    return ev


def question_features(rows: dict, qids: list[str], runner, base_letters: dict) -> dict:
    """Static and behavior features per question, written once into the
    --save-all header so clustering scripts have them in one place. The
    round-1 majority comes from the recorded four-solver opening round."""
    feats = {}
    for q in qids:
        row = rows[q]
        r1_majority = None
        try:
            r1 = runner.run_round(q, [], [], B.MASTER_ROUNDS[0], rep=0,
                                  prompts=B.question_prompts(row))
            letters = [l for _, l in B.committed_with_round([r1], len(row["options"]))]
            r1_majority = max(Counter(letters).values()) if letters else 0
        except OffCache:
            pass
        feats[q] = {"discipline": row.get("discipline"), "field": row.get("field"),
                    "subfield": row.get("subfield"), "difficulty": row.get("difficulty"),
                    "n_options": len(row["options"]), "gold": row.get("answer_letter"),
                    "base_letter": base_letters.get(q), "r1_majority": r1_majority}
    return feats


def evolve(args) -> None:
    rng = random.Random(args.seed)
    data = json.loads(args.dataset.read_text())
    if args.limit is not None:                 # debug runs only: the first N questions
        data = data[: args.limit]
    rows = {r["id"]: r for r in data}
    qids = sorted(rows)
    rng.shuffle(qids)
    half = len(qids) // 2
    train, dev = qids[:half], qids[half:]
    read_only = [args.cache, args.treegrow_cache]
    cap = args.max_calls_per_question
    if args.live:
        # the live runner loads its own cache file itself (and resumes from it)
        runner = BudgetedRunner(rows, read_only, lock=not args.ignore_cache_lock,
                                base_urls=args.base_urls, model=args.model,
                                temperature=args.temperature, answer_tokens=args.answer_tokens,
                                cache_path=args.live_cache, max_calls=args.max_total_calls,
                                api_key=args.api_key, progress=True)
        print(f"LIVE mode: new rounds go to {args.live_cache}; "
              f"novelty budget {args.novelty_budget} calls/candidate, "
              f"per-question cap {cap}, total cap {args.max_total_calls}")
    else:
        if args.live_cache.exists():          # rounds a live run recorded are replayable too
            read_only.append(args.live_cache)
        runner = CacheRunner(read_only)

    seeds = {"program_b": PROGRAM_B, "program_b_vote": PROGRAM_B_VOTE,
             "program_fixed": PROGRAM_FIXED, "program_fixed_vote": PROGRAM_FIXED_VOTE,
             **PROBES}
    if getattr(args, "no_eliminator", False):
        seeds["program_b"] = PROGRAM_B_NOELIM
        seeds["program_b_vote"] = PROGRAM_B_NOELIM_VOTE
        seeds = {k: p for k, p in seeds.items()
                 if all(r["do"] in ACTIONS or r["do"] == "continue" or r["do"].startswith("stop:")
                        for r in p["rules"])}
        print(f"eliminator removed: {len(ACTIONS)} moves available, "
              f"{len(seeds)} seed programs")
    scores: dict[str, Eval] = {}        # canon -> train Eval
    programs: dict[str, dict] = {}
    live_spent: dict[str, int] = {}     # canon -> novel calls it paid for
    population: list = []

    def cutoff_acc() -> float:
        """Train accuracy needed to enter the population right now."""
        if len(population) < args.population:
            return 0.0
        return population[args.population - 1][0] / len(train)

    def score(prog: dict) -> Eval:
        key = canon_prog(prog)
        if key in scores:
            return scores[key]
        if args.live:
            runner.reset_budget(0)             # pass 1: existing cache only, free
        ev = eval_program(prog, runner, rows, train, args.rep, max_calls=cap)
        spent = 0
        if args.live and ev.marks.count("x"):
            gaps = [i for i, m in enumerate(ev.marks) if m == "x"]
            covered_acc = ev.correct / ev.covered if ev.covered else 0.0
            mean_calls = (sum(ev.calls[i] for i in range(len(train)) if ev.marks[i] != "x")
                          / ev.covered) if ev.covered else cap
            # How many gaps the budget can afford, at this program's usual
            # cost per question. Gaps beyond that stay off-cache (= wrong), so
            # the prescreen asks whether the program could clear the cutoff
            # even after the affordable fills, not just on what it covers.
            fillable = min(len(gaps), int(args.novelty_budget // max(mean_calls, 1.0)))
            expected = (ev.correct + covered_acc * fillable) / len(train)
            passes = (fillable > 0
                      and ev.covered >= args.min_coverage * len(train)
                      and expected >= cutoff_acc() - args.prescreen_margin)
            if passes:
                # Admit exactly `fillable` gaps (train order is already a seeded
                # shuffle, so this is a random subset) and let each run to
                # completion: no question is cut off mid-way with its calls
                # wasted, and the admitted set is deterministic. The shared
                # budget stays on as a rail, with room for the last admitted
                # question to finish.
                runner.reset_budget(args.novelty_budget + cap)
                eval_program(prog, runner, rows, train, args.rep, max_calls=cap,
                             workers=args.workers, only=gaps[:fillable], into=ev)
                spent = runner.novel_calls
                runner.reset_budget(0)
        scores[key], programs[key], live_spent[key] = ev, prog, spent
        return ev

    t_start = time.perf_counter()
    population = [(score(p).correct, name, p) for name, p in seeds.items()]
    population.sort(key=lambda t: -t[0])
    print(f"\ntrain {len(train)} q, dev {len(dev)} q, "
          f"population {args.population}, {args.generations} generations")

    try:
        for gen in range(1, args.generations + 1):
            if args.live:
                runner.stage(f"gen {gen}/{args.generations}")
            cands = []
            seen_this_gen: set[str] = set()   # two parents can produce the same child
            for _, name, parent in population[:args.population]:
                for _ in range(args.offspring):
                    child = mutate_program(parent, rng)
                    try:
                        validate_program(child)
                    except (AssertionError, ValueError):
                        continue
                    key = canon_prog(child)
                    if key in scores or key in seen_this_gen:
                        continue
                    seen_this_gen.add(key)
                    cands.append((name, child))
            offspring = [(score(child).correct, name, child)
                         for name, child in tqdm(cands, desc=f"gen {gen}",
                                                 unit="prog", leave=False)]
            population = sorted(population + offspring, key=lambda t: -t[0])[:args.population]
            best_c, best_name, best = population[0]
            ev = scores[canon_prog(best)]
            live_note = (f", live calls {getattr(runner, 'calls', 0)}"
                         f" (follow-ups {getattr(runner, 'followups', 0)}, "
                         f"summaries {getattr(runner, 'summaries', 0)})" if args.live else "")
            print(f"gen {gen:>3}: best train {best_c / len(train):.2%} "
                  f"(covered {ev.covered / len(train):.0%}, lineage {best_name}), "
                  f"{len(offspring)} new, {len(scores)} evaluated{live_note}")
    except SF.BudgetExhausted as exc:
        print(f"\nstopped early: {exc} -- reporting what was scored so far")
    except KeyboardInterrupt:
        print("\ninterrupted -- reporting what was scored so far "
              "(recorded rounds are already on disk and will be reused)")

    # The fresh recheck can run on a fixed random subset of dev (--recheck-n):
    # the same questions for every program, so the comparison stays paired.
    recheck_qids = list(dev)
    if args.recheck_n is not None and args.recheck_n < len(dev):
        recheck_qids = random.Random(args.seed + 1).sample(dev, args.recheck_n)
    if args.live and args.recheck_top > 0:
        elapsed = max(time.perf_counter() - t_start, 1.0)
        rate = runner.calls / elapsed if runner.calls else None
        n_rc = len(recheck_qids)
        est = n_rc * sum((scores[canon_prog(p)].calls and
                          sum(scores[canon_prog(p)].calls) / len(train)) or 6
                         for _, _, p in population[:args.recheck_top])
        if args.recheck_baselines:
            est += n_rc * (8.2 + 8.0 + 2.0)
        note = f" (~{est / rate / 3600:.1f} h at the {rate:.2f} calls/s seen so far)" if rate else ""
        print(f"\nsearch phase done: {runner.calls} live calls. Final phase: fresh recheck "
              f"on {n_rc} dev questions, about {est:,.0f} calls{note}")

    print("\ntop 5 by train, scored on dev (selection never saw dev):")
    out_rows: list[dict] = []
    baselines_fresh: dict[str, dict] = {}

    def fresh_dev(prog: dict) -> Eval:
        """Every round re-sampled at the recheck replicate: no recorded round
        from the search is reused, so one lucky new round cannot carry it."""
        runner.reset_budget(None)
        fresh = eval_program(prog, runner, rows, recheck_qids, args.recheck_rep,
                             max_calls=cap, workers=args.workers)
        runner.reset_budget(0)
        return fresh

    try:
        for rank, (train_c, name, prog) in enumerate(population[:5]):
            if args.live:
                # dev at the recorded replicate, gaps filled live: nothing is
                # selected on dev, so paying for its off-cache rounds is fair,
                # and it keeps a program's novel branches from scoring as wrong.
                runner.reset_budget(None)
            dev_ev = eval_program(prog, runner, rows, dev, args.rep, max_calls=cap,
                                  workers=args.workers if args.live else 1)
            row = {"lineage": name, "train_acc": train_c / len(train),
                   "dev_acc": dev_ev.correct / len(dev),
                   "dev_covered": dev_ev.covered / len(dev), "program": prog,
                   "live_calls_spent": live_spent.get(canon_prog(prog), 0)}
            row.update(dev_ev.to_json("dev"))
            line = (f"  train {train_c / len(train):.2%}  dev {dev_ev.correct / len(dev):.2%} "
                    f"(dev covered {dev_ev.covered / len(dev):.0%})  lineage {name}")
            if args.live and rank < args.recheck_top:
                fresh = fresh_dev(prog)
                row["dev_fresh_acc"] = fresh.correct / len(recheck_qids)
                row["dev_fresh_rep"] = args.recheck_rep
                row["dev_fresh_n"] = len(recheck_qids)
                row.update(fresh.to_json("dev_fresh"))
                line += (f"  dev FRESH (rep {args.recheck_rep}, n={len(recheck_qids)}) "
                         f"{fresh.correct / len(recheck_qids):.2%}")
            print(line)
            out_rows.append(row)

        if args.live and args.recheck_top > 0 and args.recheck_baselines:
            # The fresh numbers need a fresh comparator, paired on the same
            # replicate, or they cannot be read against anything.
            print(f"\nreference programs on dev, same fresh replicate {args.recheck_rep}:")
            # Must match the action menu this run is using: with --no-eliminator
            # the original program_b cannot run at all.
            noelim = getattr(args, "no_eliminator", False)
            refs = [("program_b", PROGRAM_B_NOELIM if noelim else PROGRAM_B),
                    ("program_b_vote", PROGRAM_B_NOELIM_VOTE if noelim else PROGRAM_B_VOTE),
                    ("program_fixed", PROGRAM_FIXED),
                    ("solver_critic", PROBES["solver_critic"])]
            for name, prog in refs:
                try:
                    validate_program(prog)
                except (AssertionError, ValueError) as exc:
                    print(f"  {name:14} skipped: not runnable under this action menu ({exc})")
                    continue
                fresh = fresh_dev(prog)
                baselines_fresh[name] = {"dev_fresh_acc": fresh.correct / len(recheck_qids),
                                         "dev_fresh_n": len(recheck_qids),
                                         **fresh.to_json("dev_fresh")}
                print(f"  {name:14} dev FRESH {fresh.correct / len(recheck_qids):.2%}")
    except (SF.BudgetExhausted, KeyboardInterrupt) as exc:
        print(f"\nstopped during the final evaluation ({exc!r}) -- "
              f"writing what was completed")
    except Exception:
        # Never lose a multi-hour search to a fault in the reporting phase.
        # Whatever was computed still gets written below.
        import traceback
        print("\nERROR during the final evaluation -- writing what was completed:")
        traceback.print_exc()

    if args.save_all:
        base = SF.load_base_letters(args.bestofn_cache) if args.bestofn_cache.exists() else {}
        if args.live:
            runner.reset_budget(0)
        header = {"train_qids": train, "dev_qids": dev, "seed": args.seed,
                  "live": args.live,
                  "questions": question_features(rows, qids, runner, base)}
        args.save_all.parent.mkdir(parents=True, exist_ok=True)
        with args.save_all.open("w") as fh:
            fh.write(json.dumps(header) + "\n")
            for key, ev in scores.items():
                rec = {"program": programs[key], "live_calls_spent": live_spent.get(key, 0)}
                rec.update(ev.to_json("train"))
                fh.write(json.dumps(rec) + "\n")
        print(f"all {len(scores)} scored programs -> {args.save_all}")
    args.evolved_out.parent.mkdir(parents=True, exist_ok=True)
    args.evolved_out.write_text(json.dumps(
        {"n_train": len(train), "n_dev": len(dev), "seed": args.seed,
         "generations": args.generations, "evaluated": len(scores), "live": args.live,
         "live_calls_total": getattr(runner, "calls", 0) if args.live else 0,
         "v2": getattr(args, "v2", False), "answer_tokens": getattr(args, "answer_tokens", None),
         "v2_stats": dict(D.V2_STATS) if getattr(args, "v2", False) else None,
         "baselines": {"program_b": 0.1482, "program_fixed": 0.1449},
         "baselines_fresh": baselines_fresh,
         "recheck_qids": recheck_qids if args.live else [],
         "top": out_rows}, indent=2))
    print(f"saved -> {args.evolved_out}")
    if args.live:
        runner.close()


# --- validation gate --------------------------------------------------------

def validate(args) -> None:
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    records = [json.loads(l) for l in args.records.open() if l.strip()]
    runner = CacheRunner([args.cache])
    validate_program(PROGRAM_B)
    validate_program(PROGRAM_FIXED)

    mismatch, off_cache, correct = [], 0, 0
    for rec in tqdm(records, desc="replay program B", unit="q", leave=False):
        try:
            out = run_program(PROGRAM_B, runner, rows[rec["qid"]], rep=args.rep)
        except OffCache:
            off_cache += 1
            continue
        correct += out["correct"]
        track, closer = derive_track(out["actions"], len(PROGRAM_B["plan"]))
        if (out["letter"], track, closer) != (rec["letter"], rec["track"], rec["closer"]):
            mismatch.append((rec["qid"],
                             (out["letter"], track, closer),
                             (rec["letter"], rec["track"], rec["closer"])))

    n = len(records)
    rec_acc = sum(r["correct"] for r in records) / n
    print(f"\nPROGRAM_B replay:   {correct / n:.2%} ({correct}/{n})")
    print(f"recorded adaptive:  {rec_acc:.2%}")
    print(f"off-cache questions: {off_cache}")
    print(f"per-question mismatches (letter/track/closer): {len(mismatch)}")
    for qid, got, want in mismatch[:10]:
        print(f"  {qid}: replay={got}  recorded={want}")

    fx_correct, fx_off = 0, 0
    for rec in tqdm(records, desc="replay fixed recipe", unit="q", leave=False):
        try:
            out = run_program(PROGRAM_FIXED, runner, rows[rec["qid"]], rep=args.rep)
        except OffCache:
            fx_off += 1
            continue
        fx_correct += out["correct"]
    summary = json.loads(args.summary.read_text())
    print(f"\nPROGRAM_FIXED replay: {fx_correct / n:.2%} ({fx_correct}/{n}), "
          f"off-cache {fx_off}")
    print(f"recorded fixed:       {summary['fixed_recipe']['accuracy']:.2%}")

    ok = (not mismatch and off_cache == 0
          and fx_correct == summary["fixed_recipe"]["n_correct"])
    print("\nVALIDATION " + ("PASSED" if ok else "FAILED"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="Replay experiment B's rules; must match its records exactly.")
    ap.add_argument("--coverage", action="store_true",
                    help="Replay probe programs against the treegrow cache.")
    ap.add_argument("--evolve", action="store_true",
                    help="Evolve programs by random rule edits, scored by replay.")
    ap.add_argument("--treegrow-cache", type=Path,
                    default=Path("outputs/treegrow_rounds_cache.jsonl"))
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--offspring", type=int, default=3,
                    help="Mutations tried per surviving program per generation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evolved-out", type=Path,
                    default=Path("outputs/program_mcq_evolved.json"))
    ap.add_argument("--save-all", type=Path, default=None,
                    help="Also write every scored program with its per-question "
                         "train outcomes, letters, calls and action paths to this "
                         "jsonl (first line: the split and per-question features).")
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_strict_train.json"))
    ap.add_argument("--records", type=Path, default=Path("outputs/adaptive_debate_records.jsonl"))
    ap.add_argument("--summary", type=Path, default=Path("outputs/adaptive_debate_summary.json"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/adaptive_rounds_cache.jsonl"))
    ap.add_argument("--bestofn-cache", type=Path, default=Path("outputs/bestofn_strict_cache.jsonl"),
                    help="Source of each question's base letter for the --save-all header.")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug only: use just the first N questions of the dataset "
                         "(before the train/dev split). Never for a real run.")
    ap.add_argument("--no-eliminator", action="store_true",
                    help="Drop the eliminator moves (eliminate, elim_blind) and use "
                         "eliminator-free seeds. It cannot be justified in a deployed "
                         "system and its survivor list rarely reached the next persona.")
    ap.add_argument("--digest-head", type=int, default=700,
                    help="Characters of each prior response a later persona sees, from "
                         "the start (default 700, the window every 14B recording used).")
    ap.add_argument("--digest-tail", type=int, default=0,
                    help="Characters shown from the END of each prior response as well. "
                         "The committed conclusion sits at ~99%% of the text, so a tail "
                         "is what carries it. Non-default windows change the round cache "
                         "key, so they can never replay 700/0 recordings by mistake.")
    ap.add_argument("--commit-followup", action="store_true",
                    help="When a reply commits no letter (usually cut off at the token "
                         "cap), send one short continuation asking it to commit now. "
                         "Changes the round cache key, so it never replays recordings "
                         "made without it.")
    ap.add_argument("--v2", action="store_true",
                    help="v2 pipeline: careful-reasoning prompts, 6144-token replies, and a "
                         "summary follow-up whose text is what later personas read. Changes "
                         "the round cache key (v=2). Needs a digest tail; replaces "
                         "--commit-followup.")
    ap.add_argument("--max-calls-per-question", type=int, default=16,
                    help="A program that would exceed this many calls on one question is "
                         "stopped instead (never binds on the hand-written programs).")
    # --- live mode: execute off-cache rounds instead of counting them wrong ---
    live = ap.add_argument_group("live mode (needs vLLM)")
    live.add_argument("--live", action="store_true",
                      help="Run off-cache rounds on the model and cache them.")
    live.add_argument("--live-cache", type=Path, default=Path("outputs/program_live_rounds_cache.jsonl"),
                      help="Where NEW rounds are written. The treegrow/adaptive caches are read-only.")
    live.add_argument("--base-urls", default="http://localhost:7472/v1")
    live.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    live.add_argument("--api-key", default="EMPTY")
    live.add_argument("--temperature", type=float, default=0.7)
    live.add_argument("--answer-tokens", type=int, default=None,
                      help="Output budget per reasoning call (default 3072; 6144 under --v2).")
    live.add_argument("--novelty-budget", type=int, default=300,
                      help="New model calls a candidate may spend filling its off-cache "
                           "questions; beyond this they count as wrong again.")
    live.add_argument("--prescreen-margin", type=float, default=0.02,
                      help="A candidate spends live calls only if its accuracy on the "
                           "covered questions is within this of the population cutoff.")
    live.add_argument("--min-coverage", type=float, default=0.5,
                      help="...and only if the existing cache covers at least this "
                           "fraction of the train batch for it.")
    live.add_argument("--workers", type=int, default=32,
                      help="Questions run in parallel during live fills and rechecks.")
    live.add_argument("--recheck-top", type=int, default=3,
                      help="How many final programs get a fresh-replicate dev run.")
    live.add_argument("--recheck-rep", type=int, default=1,
                      help="Replicate number for the fresh dev run (0 = the recorded one).")
    live.add_argument("--recheck-n", type=int, default=None,
                      help="Fresh-recheck only this many dev questions (a fixed random "
                           "subset, the same for every program). Default: all of dev.")
    live.add_argument("--no-recheck-baselines", dest="recheck_baselines",
                      action="store_false", default=True,
                      help="Skip the fresh dev run of program B, the fixed recipe and "
                           "solver->critic (about 3 x 1,515 x 6 calls).")
    live.add_argument("--max-total-calls", type=int, default=400_000,
                      help="Safety rail: the run stops and reports when this is reached.")
    live.add_argument("--ignore-cache-lock", action="store_true",
                      help="Open the live cache even if its lock file exists (only when the "
                           "run that made it is known to be dead).")
    args = ap.parse_args()
    if args.v2 and args.commit_followup:
        raise SystemExit("--v2 already commits through its summary call; drop --commit-followup")
    if args.v2 and args.digest_tail <= 0:
        raise SystemExit("--v2 needs --digest-tail > 0 (e.g. --digest-head 300 --digest-tail 900)")
    if args.answer_tokens is None:
        args.answer_tokens = 6144 if args.v2 else 3072
    if (args.digest_head, args.digest_tail) != (700, 0):
        D.set_digest(args.digest_head, args.digest_tail)
        print(f"digest window: {args.digest_head} head + {args.digest_tail} tail chars "
              f"(non-default: round cache keys differ from the 700/0 recordings)")
    if getattr(args, "no_eliminator", False):
        drop_eliminator()
    if args.commit_followup:
        D.set_commit_followup(True)
        print("commit follow-up ON (round cache keys carry c=1)")
    if args.v2:
        SF.set_v2(True)
        print(f"v2 pipeline ON: careful-reasoning prompts, {args.answer_tokens}-token replies, "
              f"summary follow-up (round cache keys carry v=2)")
    if args.evolve:
        # A command pasted across two lines silently loses its trailing flags.
        # So: never lose the per-program log in live mode, and show every
        # output path before anything is loaded or spent.
        if args.live and args.save_all is None:
            args.save_all = args.evolved_out.with_name(args.evolved_out.stem + "_all.jsonl")
        print(f"results   -> {args.evolved_out}"
              + ("  [exists, will be overwritten]" if args.evolved_out.exists() else ""))
        if args.save_all:
            print(f"all progs -> {args.save_all}"
                  + ("  [exists, will be overwritten]" if args.save_all.exists() else ""))
        if args.live:
            print(f"live cache-> {args.live_cache}  (shared with no other running process)")
    if args.validate:
        validate(args)
    if args.coverage:
        coverage(args)
    if args.evolve:
        evolve(args)
    if not (args.validate or args.coverage or args.evolve):
        ap.print_help()

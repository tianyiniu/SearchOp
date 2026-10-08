"""The program space for the per-group search: what a program is, how one is
sampled, edited and compared, and how its per-question results are kept.

No model calls here. Everything a search or a seed stage needs that is a pure
function of programs and their recorded results lives in this module, so the
seed stage (program_seeds.py) and the search (evolve_program_clusters.py)
agree by construction.

A program is the same object evolve_program_mcq interprets:

    {"plan": [round spec, ...],            the opening rounds, run by "continue"
     "rules": [{"when": [cond, ...], "do": action}, ...],
     "default": "stop:<read>"}

The rule grammar (conditions, actions, stop reads) is evolve_program_mcq's;
this module adds the plan grammar (which opening rounds a program may have),
a plan mutation, a random program sampler, the structural and semantic
distances used for diversity, and the per-question score record.

Executor settings for the whole new pipeline are fixed in `configure_executor`:
v2 (careful-reasoning prompts, 6144-token replies, a summary follow-up whose
text is what later speakers read), a 300+900 character digest window, and no
eliminator moves. Every recording made under these settings carries them in
its cache key, so recordings made under the old settings are never replayed.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import debate_mcq as D  # noqa: E402
import schema_fitness as SF  # noqa: E402
import evolve_program_mcq as M  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# --- executor settings (the context-management changes) --------------------

DIGEST_HEAD, DIGEST_TAIL = 300, 900
ANSWER_TOKENS = 6144
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
DEFAULT_BASE_URLS = "http://localhost:7472/v1"


def configure_executor(digest_head: int | None = None, digest_tail: int | None = None,
                       visible_reasoning: bool = False,
                       summary_words: int | None = None, executor: str = "v2",
                       window: int | None = None, answers: str = "letters",
                       judge_model: str | None = None, judge_cache: Path | None = None,
                       high_cost: int | None = None, turn_cap: int | None = None,
                       judge_persona: bool = False, any_round_width: bool = False,
                       last_round_vote: bool = False, plain_instruction: bool = False,
                       count_read_summaries: bool = False, total_cap: int | None = None) -> dict:
    """Switch the shared executor to the new pipeline's settings. Returns the
    settings as a dict, which the archive header records and a resume checks.
    `visible_reasoning` is for models that think in a hidden channel (gpt-oss);
    the settings dict names it only when on, so archives made without it still
    match on resume. `executor="v3"` is the effort-and-visibility executor and
    grammar (see configure_v3); the default, v2, is unchanged. A digest or
    summary length left as None takes the executor's default. `answers="open"`
    (v3 only) is for open-answer datasets (HLE): see configure_v3. So are `high_cost`,
    `turn_cap`, `judge_persona` and `last_round_vote` (None / False: the defaults, 5 and 16
    turns, no judge, the last-commit reads of every earlier run)."""
    if executor == "v3":
        return configure_v3(digest_head, digest_tail, visible_reasoning, summary_words, window,
                            answers=answers, judge_model=judge_model, judge_cache=judge_cache,
                            high_cost=high_cost, turn_cap=turn_cap, judge_persona=judge_persona,
                            any_round_width=any_round_width, last_round_vote=last_round_vote,
                            plain_instruction=plain_instruction,
                            count_read_summaries=count_read_summaries, total_cap=total_cap)
    if answers != "letters":
        raise SystemExit("open answers need the v3 executor")
    if (high_cost is not None or turn_cap is not None or judge_persona or any_round_width or last_round_vote
            or plain_instruction or count_read_summaries or total_cap is not None):
        raise SystemExit("--high-cost, --turn-cap, --total-cap, --judge-persona, --any-round-width, "
                         "--last-round-vote, --plain-instruction and --count-read-summaries need the v3 executor")
    global MAX_TURNS, RUN_CAP, ANY_ROUND_WIDTH  # the v3 options back at their defaults (not used by v2)
    MAX_TURNS = RUN_CAP = DEFAULT_TURN_CAP
    ANY_ROUND_WIDTH = False
    M.set_last_round_vote(False)
    D.set_summary_split(False)
    M.set_count_read_summaries(False)
    D.set_high_cost(D.DEFAULT_HIGH_COST)
    if D.JUDGE_ON:
        D.set_judge_persona(False)
    if executor != "v2":
        raise SystemExit(f"unknown executor {executor!r}")
    digest_head = DIGEST_HEAD if digest_head is None else digest_head
    digest_tail = DIGEST_TAIL if digest_tail is None else digest_tail
    summary_words = 120 if summary_words is None else summary_words
    if digest_tail <= 0:
        raise SystemExit("the v2 executor needs a digest tail > 0")
    D.set_digest(digest_head, digest_tail)
    SF.set_v2(True)
    M.drop_eliminator()
    D.set_round_parallel(True)        # speakers of one round in flight together
    settings = {"v2": True, "digest": [digest_head, digest_tail], "answer_tokens": ANSWER_TOKENS,
                "eliminator": False}
    SF.set_visible_reasoning(visible_reasoning)      # on or off: a later configure never inherits it
    if visible_reasoning:
        settings["visible_reasoning"] = True
    # Both are named in the settings only when on, like visible_reasoning, so
    # archives made without them still match on resume.
    if summary_words != 120:
        D.set_summary_words(summary_words)
        settings["summary_words"] = summary_words
    settings["prompts"] = D.prompt_signature()        # the texts speakers are sent (since 2026-10-06)
    return settings


def seed_protocols() -> dict[str, dict]:
    """The literature programs used as seeds, in their fixed order."""
    return dict(PROTOCOLS)


# v3 defaults: a 500-word summary, and a digest window that holds a 500-word reply whole
V3_DIGEST = (2000, 2000)
V3_SUMMARY_WORDS = 500


JUDGE = None          # the judge of open answers, once configure_v3 has made it (its .stats count calls)


def configure_v3(digest_head: int | None, digest_tail: int | None, visible_reasoning: bool,
                 summary_words: int | None, window: int | None,
                 answers: str = "letters", judge_model: str | None = None,
                 judge_cache: Path | None = None, high_cost: int | None = None,
                 turn_cap: int | None = None, judge_persona: bool = False,
                 any_round_width: bool = False, last_round_vote: bool = False,
                 plain_instruction: bool = False, count_read_summaries: bool = False,
                 total_cap: int | None = None) -> dict:
    """The v3 executor (debate_mcq, 'v3') and the v3 grammar: every round is
    speakers x effort (low/high) x visibility (sees the debate / blind), the same
    names serve as plan rounds and as moves, and the literature programs are
    rewritten in them (plus their high-effort variants). With answers="open"
    (HLE): speakers commit 'ANSWER: <answer>' in a fixed format, answers are
    compared normalised (debate_mcq.set_open_answers), and a final answer is
    graded by the judge model (judge_answers.Judge, cached in `judge_cache`).
    The settings then name the answer mode and the judge; otherwise they are
    exactly what they were, so earlier archives still match on resume.

    Budget and judge speaker (both off by default, so earlier runs are unchanged):
    `high_cost` is the turns one high-effort speaker counts as (default 5, always in the
    settings as "high_cost"); `turn_cap` is the most turns one question may use and one
    round may cost (default 16; named in the settings as "turn_cap" only when not 16);
    `judge_persona` adds the judge speaker (debate_mcq: it chooses among the answers
    committed so far) as plan rounds and moves, and the judge seeds (judge_seeds; named
    in the settings as "judge_persona" only when on). `any_round_width` makes the width
    edit (set_width in place of plan_width) change the number of solvers of any solver
    round, not only the first plan round (named in the settings as "any_round_width" only
    when on). `last_round_vote` (v4) makes the stop reads last_commit and last_speaker read
    the most common answer of the last round that committed one, not the last speaker's
    (evolve_program_mcq.LAST_ROUND_VOTE; named in the settings as "last_round_vote" only
    when on). None of them is in any cache key: they change what is read off a recorded
    debate, never the debate. `plain_instruction` (off by default) asks every speaker only for
    the commitment line, without the v2 request to think carefully and take the space it
    needs, so the effort sent with the request alone sets how long it thinks; it changes
    every prompt, so it is in the cache key ("i") and named in the settings only when on.
    `count_read_summaries` (off by default) records a speaker's answer-locked summary tokens apart and leaves them out of a
    debate's tokens when no later round read them (evolve_program_mcq.COUNT_READ_SUMMARIES); it
    changes no prompt and no answer, only the token count, and it is named in the settings only when on.
    The settings always name the prompts' signature (debate_mcq.prompt_signature, since 2026-10-06), so
    an archive made with other prompts is refused."""
    global JUDGE
    if not window:
        raise SystemExit("the v3 executor needs the server's context window")
    head, tail = (V3_DIGEST[0] if digest_head is None else digest_head,
                  V3_DIGEST[1] if digest_tail is None else digest_tail)
    words = V3_SUMMARY_WORDS if summary_words is None else summary_words
    global MAX_TURNS, RUN_CAP, ANY_ROUND_WIDTH
    ANY_ROUND_WIDTH = bool(any_round_width)
    D.set_high_cost(D.DEFAULT_HIGH_COST if high_cost is None else high_cost)
    MAX_TURNS = DEFAULT_TURN_CAP if turn_cap is None else int(turn_cap)
    if MAX_TURNS < 1:
        raise SystemExit(f"--turn-cap must be at least 1, not {MAX_TURNS}")
    RUN_CAP = MAX_TURNS if total_cap is None else int(total_cap)
    if RUN_CAP < MAX_TURNS:
        raise SystemExit(f"--total-cap {RUN_CAP} is below the turn cap {MAX_TURNS}")
    D.set_judge_persona(judge_persona)
    M.set_last_round_vote(last_round_vote)
    D.set_digest(head, tail)
    SF.set_v3(True, window)
    M.drop_eliminator()
    D.set_round_parallel(True)
    D.set_summary_words(words)
    SF.set_visible_reasoning(visible_reasoning)      # on or off: a later configure never inherits it
    SF.set_plain_instruction(plain_instruction)
    if answers not in ("letters", "open", "math"):
        raise SystemExit(f"unknown answer mode {answers!r}")
    SF.set_math_answers(answers == "math")         # MATH answers are open answers read as math (2026-10-07)
    SF.set_open_answers(answers in ("open", "math"))
    D.set_summary_split(count_read_summaries)
    M.set_count_read_summaries(count_read_summaries)
    use_v3_grammar()
    settings = {"executor": "v3", "digest": [head, tail], "summary_words": words, "window": int(window),
                "high_cost": D.HIGH_COST, "eliminator": False, "prompts": D.prompt_signature()}
    if MAX_TURNS != DEFAULT_TURN_CAP:
        settings["turn_cap"] = MAX_TURNS
    if total_cap is not None:                        # named only when given, so earlier archives still match
        settings["total_cap"] = RUN_CAP
    if judge_persona:
        settings["judge_persona"] = True
    if any_round_width:
        settings["any_round_width"] = True
    if last_round_vote:
        settings["last_round_vote"] = True
    if visible_reasoning:
        settings["visible_reasoning"] = True
    if plain_instruction:
        settings["plain_instruction"] = True
    if count_read_summaries:
        settings["count_read_summaries"] = True
    if answers == "open":
        import judge_answers as J
        JUDGE = J.Judge(judge_cache, model=judge_model or J.JUDGE_MODEL)
        M.set_grader(JUDGE.grade_row)
        settings["answers"] = "open"
        settings["judge"] = JUDGE.settings()
    elif answers == "math":                         # graded by math-verify, as the external baselines are
        import math_answers as MA
        JUDGE = None
        M.set_grader(MA.grade_row)
        settings["answers"] = "math"
    else:
        JUDGE = None
        M.set_grader(None)
    return settings


def check_rows(rows) -> None:
    """Stop unless the dataset suits the answer mode: open answers need each
    row's key in 'answer'; letters need 'options' and 'answer_letter' (a row
    without options would never commit a letter)."""
    rows = list(rows)
    if D.OPEN:
        bad = [r["id"] for r in rows if not str(r.get("answer") or "").strip()]
        if bad:
            raise SystemExit(f"--answers open: {len(bad)} questions have no 'answer' (e.g. {bad[0]})")
    else:
        bad = [r["id"] for r in rows if not r.get("options") or r.get("answer_letter") is None]
        if bad:
            raise SystemExit(f"{len(bad)} questions have no options or answer_letter (e.g. {bad[0]}); "
                             f"an open-answer dataset (HLE) needs --answers open")


def add_executor_args(ap, default_executor: str = "v3") -> None:
    """The executor options, the same on every cluster-pipeline entry script."""
    ap.add_argument("--executor", choices=["v2", "v3"], default=default_executor,
                    help="v3: effort and visibility per round, no reply cap, model-card sampling, "
                         "answer-locked summaries. v2: the executor of the earlier runs")
    ap.add_argument("--digest-head", type=int, default=None,
                    help=f"characters shown from the start of a reply shown without a summary "
                         f"(default {V3_DIGEST[0]} under v3, {DIGEST_HEAD} under v2)")
    ap.add_argument("--digest-tail", type=int, default=None,
                    help=f"... and from its end (default {V3_DIGEST[1]} under v3, {DIGEST_TAIL} under v2)")
    ap.add_argument("--visible-reasoning", action="store_true",
                    help="for models that think in a hidden channel (gpt-oss): ask every speaker to "
                         "write its reasoning in the visible reply; changes prompts and cache keys")
    ap.add_argument("--summary-words", type=int, default=None,
                    help=f"word limit of the summary later speakers read (default {V3_SUMMARY_WORDS} under "
                         f"v3, 120 under v2); a committed reply within it is shown as it is")
    ap.add_argument("--context-window", type=int, default=None,
                    help="v3: the server's context window (default: read from the server's /models)")
    ap.add_argument("--answers", choices=["letters", "open", "math"], default="letters",
                    help="letters: multiple choice with an options list (SuperGPQA). open: open answers "
                         "(HLE), committed in a fixed format, compared normalised, graded by a judge model. "
                         "math: open answers to MATH questions, given in \\boxed{} (or an ANSWER line), "
                         "compared and graded by math-verify (math_answers.py); no judge")
    ap.add_argument("--judge-model", default=None,
                    help="open answers: the judge (default judge_answers.JUDGE_MODEL, gpt-6-luna)")
    ap.add_argument("--judge-cache", type=Path, default=None,
                    help="open answers: the verdict cache (default outputs/judge_cache/<model>_<effort>_<prompt>"
                         ".jsonl, shared by the search, the evaluation and the baselines)")
    ap.add_argument("--high-cost", type=int, default=None,
                    help="v3: the turns one high-effort speaker counts as (default 5)")
    ap.add_argument("--turn-cap", type=int, default=None,
                    help=f"v3: the most turns one question may use and one round may cost (default "
                         f"{DEFAULT_TURN_CAP}); --max-calls-per-question follows it")
    ap.add_argument("--total-cap", type=int, default=None,
                    help="v3: the most turns one question's whole debate may use (default: the turn cap). "
                         "The turn cap then bounds only the plan (its opening rounds) and one round; rules "
                         "may run extra rounds up to this cap, and a round that would pass it is not run "
                         "(the default stop decides). --max-calls-per-question follows it")
    ap.add_argument("--any-round-width", action="store_true",
                    help="v3: the width edit (set_width, in place of plan_width) changes the number of "
                         "solvers of any solver round (the first, a later plan round, or an extra round "
                         "run by a rule), keeping its effort and visibility")
    ap.add_argument("--judge-persona", action="store_true",
                    help="v3: add the judge speaker (chooses among the answers committed so far) as "
                         "plan rounds and moves, and the judge seed programs")
    ap.add_argument("--last-round-vote", action="store_true",
                    help="v3 (the global pipeline): the stops last_commit and last_speaker read the most common "
                         "answer of the last round that committed one, not the last speaker's; changes no "
                         "recording, only the answer read off it")
    ap.add_argument("--plain-instruction", action="store_true",
                    help="v3: ask every speaker only for the 'ANSWER: <letter>' line, without the request to "
                         "think carefully and take the space it needs (the effort sent with the request sets "
                         "how long it thinks); changes prompts and cache keys")
    ap.add_argument("--count-read-summaries", action="store_true",
                    help="v3: a debate's tokens leave out the summary of a speaker that no later speaker read "
                         "(it is still written, and it never changes an answer); changes only the token count")


def server_window(base_urls: str, model: str, api_key: str = "EMPTY") -> int | None:
    """The smallest max_model_len the servers report for `model`, or None if
    none of them can be asked."""
    import urllib.request
    found = []
    for url in [u.strip().rstrip("/") for u in base_urls.split(",") if u.strip()]:
        try:
            req = urllib.request.Request(url + "/models", headers={"Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=10) as fh:
                data = json.loads(fh.read())
        except Exception:
            continue
        found += [m["max_model_len"] for m in data.get("data", [])
                  if m.get("id") == model and isinstance(m.get("max_model_len"), int)]
    return min(found) if found else None


def configure_from_args(args, fallback_window: int | None = None) -> dict:
    """configure_executor from add_executor_args' options. Under v3 the window
    is --context-window, else the server's, else `fallback_window`."""
    budget = {"high_cost": getattr(args, "high_cost", None), "turn_cap": getattr(args, "turn_cap", None),
              "judge_persona": getattr(args, "judge_persona", False),
              "any_round_width": getattr(args, "any_round_width", False),
              "last_round_vote": getattr(args, "last_round_vote", False),
              "plain_instruction": getattr(args, "plain_instruction", False),
              "count_read_summaries": getattr(args, "count_read_summaries", False),
              "total_cap": getattr(args, "total_cap", None)}
    if args.executor == "v2":
        settings = configure_executor(DIGEST_HEAD if args.digest_head is None else args.digest_head,
                                      DIGEST_TAIL if args.digest_tail is None else args.digest_tail,
                                      args.visible_reasoning,
                                      summary_words=120 if args.summary_words is None else args.summary_words,
                                      **budget)
        if getattr(args, "max_calls_per_question", 0) is None:
            args.max_calls_per_question = DEFAULT_TURN_CAP
        return settings
    window = args.context_window or server_window(args.base_urls, args.model, getattr(args, "api_key", "EMPTY"))
    if window is None:
        window = fallback_window
    if window is None:
        raise SystemExit(f"could not read the context window of {args.model} from {args.base_urls}; "
                         f"is the server up? (or pass --context-window)")
    settings = configure_executor(args.digest_head, args.digest_tail, args.visible_reasoning,
                                  summary_words=args.summary_words,
                                  executor="v3", window=window, answers=args.answers,
                                  judge_model=args.judge_model, judge_cache=args.judge_cache, **budget)
    # the per-question call cap of the entry scripts is the whole-debate cap (--total-cap, else the
    # turn cap): one number, set once
    if hasattr(args, "max_calls_per_question"):
        if args.max_calls_per_question is None:
            args.max_calls_per_question = RUN_CAP
        elif args.max_calls_per_question != RUN_CAP:
            raise SystemExit(f"--max-calls-per-question {args.max_calls_per_question} differs from the debate "
                             f"cap {RUN_CAP}: set it with --turn-cap / --total-cap alone")
    return settings


# --- plan grammar ------------------------------------------------------------
# What an opening round may be. Every kind here is also a round kind the rule
# grammar can name (last_round:<kind>), so rules can react to plan rounds.

MAX_PLAN = 5
OPENING_WIDTHS = (1, 2, 3, 4)


def solvers(k: int) -> dict:
    return {"personas": ["solver"] * k}


# Specs are kept in normalised form: no "sees" key when it would be the
# default ("all"). The cache key normalises the same way, so a spec written
# either way records and replays identically; normalising here keeps the
# program text canonical too.
PLAN_ROUNDS: dict[str, dict] = {
    "solver": solvers(1), "solver_x2": solvers(2), "solver_x3": solvers(3), "solver_x4": solvers(4),
    "critic": {"personas": ["critic"]},
    "verifier": {"personas": ["verifier"]},
    "synthesizer": {"personas": ["synthesizer"]},
    "expert_blind": {"personas": ["expert"], "sees": "none"},
    "fresh_pair": {"personas": ["independent", "independent"]},
    "expert_solver": {"personas": ["expert", "solver"], "sees": "last_round"},
}


def norm_spec(spec: dict) -> dict:
    out = {"personas": list(spec["personas"])}
    if spec.get("sees", D.DEFAULT_SEES) != D.DEFAULT_SEES:
        out["sees"] = spec["sees"]
    if spec.get("effort", D.DEFAULT_EFFORT) != D.DEFAULT_EFFORT:
        out["effort"] = spec["effort"]
    return out


def normalize_program(prog: dict) -> dict:
    """The same program with every plan spec in normalised form and the rule
    list deep-copied, so the text is canonical before it is hashed. Under v3 the
    first round's visibility is dropped: it has nothing to see either way."""
    plan = [norm_spec(s) for s in prog["plan"]]
    if D.V3 and plan:
        plan[0].pop("sees", None)
    return {"plan": plan,
            "rules": [{"when": list(r["when"]), "do": r["do"]} for r in prog["rules"]],
            "default": prog["default"]}


def plan_round_name(spec: dict) -> str:
    spec = norm_spec(spec)
    for name, s in PLAN_ROUNDS.items():
        if s == spec:
            return name
    return "+".join(spec["personas"]) + (f"|{spec['sees']}" if spec.get("sees") else "")


def validate_program(prog: dict) -> None:
    """The rule grammar's validation plus the plan grammar's (and, under v3,
    the checks in validate_v3)."""
    M.validate_program(prog)
    plan = prog["plan"]
    if not (1 <= len(plan) <= MAX_PLAN):
        raise ValueError(f"plan must have 1..{MAX_PLAN} rounds, has {len(plan)}")
    for spec in plan:
        if norm_spec(spec) not in PLAN_ROUNDS.values():
            raise ValueError(f"plan round {spec} is not in the plan grammar")
    for rule in prog["rules"]:
        if not (1 <= len(rule["when"]) <= 3):
            raise ValueError(f"a rule needs 1..3 conditions: {rule}")
    if D.V3:
        validate_v3(prog)


def plan_cost(prog: dict) -> int:
    """Speaker turns the plan rounds cost when every one of them runs. Over MAX_TURNS the last
    plan round can never run: run_program stops a question at the cap, so such a program runs
    as a shorter one than it says. The seed writer rejects these plans (program_guide); the
    searches redraw such children and refuse such seeds. Not part of validate_program, so the
    archives of runs made before this check still load and replay."""
    return sum(D.spec_cost(spec) for spec in prog["plan"])


def never_runs_as_written(prog: dict) -> str | None:
    """Why `prog` can never run as it is written, or None: its plan (all of its opening rounds) costs
    more than the turn cap. Rounds its rules add are bounded only by the whole-debate cap (RUN_CAP,
    --total-cap): one that would pass it is not run, the default stop decides, and the debate is
    recorded as cut (run_program's "capped", cap_cut)."""
    if (cost := plan_cost(prog)) > MAX_TURNS:
        return f"its plan rounds cost {cost} turns, over the {MAX_TURNS}-turn cap, so its last plan round never runs"
    return None


# --- the v3 grammar: speakers x effort x visibility ------------------------------------
# One vocabulary of rounds serves as plan rounds and as moves. A round is named
# <speakers>[|high][|blind]:
#   speakers  who speaks (SPEAKERS_V3); speakers of one round never see each other
#   |high     the model's high reasoning effort; each such speaker counts HIGH_COST turns
#   |blind    the speakers see only the question (solver and expert rounds only); otherwise
#             they see every earlier reply (or its summary)
# No round may cost more than the 16-turn cap, so four high-effort solvers are not a round.

SPEAKERS_V3: dict[str, list[str]] = {
    "solver": ["solver"], "solver_x2": ["solver"] * 2, "solver_x3": ["solver"] * 3,
    "solver_x4": ["solver"] * 4, "expert": ["expert"], "expert_solver": ["expert", "solver"],
    "critic": ["critic"], "verifier": ["verifier"], "synthesizer": ["synthesizer"],
}
BLINDABLE = ("solver", "solver_x2", "solver_x3", "solver_x4", "expert", "expert_solver")
DEFAULT_TURN_CAP = 16
MAX_TURNS = DEFAULT_TURN_CAP      # the per-question turn cap, and the most one round may cost; configure_v3
                                  # sets it (--turn-cap). With --total-cap it bounds the plan (the opening
                                  # rounds) and one round only
RUN_CAP = DEFAULT_TURN_CAP        # the most turns a whole debate may use: --total-cap, else MAX_TURNS
JUDGE_SPEAKERS: dict[str, list[str]] = {D.JUDGE_PERSONA: [D.JUDGE_PERSONA]}   # with --judge-persona only


def speakers_v3() -> dict[str, list[str]]:
    """The speaker menu: SPEAKERS_V3, plus the judge when it is on (never blind: it must
    see the answers it chooses between)."""
    return {**SPEAKERS_V3, **JUDGE_SPEAKERS} if D.JUDGE_ON else dict(SPEAKERS_V3)


def v3_spec(base: str, high: bool = False, blind: bool = False) -> dict:
    spec = {"personas": list(speakers_v3()[base])}
    if blind:
        spec["sees"] = "none"
    if high:
        spec["effort"] = "high"
    return spec


def v3_name(spec: dict) -> str:
    """The v3 name of a round spec ('verifier|high', 'solver_x2|blind', ...)."""
    base = next(b for b, ps in speakers_v3().items() if ps == list(spec["personas"]))
    return (base + ("|high" if spec.get("effort") == "high" else "")
            + ("|blind" if spec.get("sees") == "none" else ""))


def v3_vocabulary() -> dict[str, dict]:
    out = {}
    for base in speakers_v3():
        for high in (False, True):
            for blind in ((False, True) if base in BLINDABLE else (False,)):
                spec = v3_spec(base, high, blind)
                if D.spec_cost(spec) <= MAX_TURNS:
                    out[v3_name(spec)] = spec
    return out


def use_v3_grammar() -> None:
    """Replace the plan rounds, the move menu and the literature programs with
    the v3 ones, for this process."""
    vocab = v3_vocabulary()
    PLAN_ROUNDS.clear()
    PLAN_ROUNDS.update({name: dict(spec) for name, spec in vocab.items()})
    M.set_actions({name: [dict(spec)] for name, spec in vocab.items()})
    PROTOCOLS.clear()
    PROTOCOLS.update(protocols_v3())


def round_slots(prog: dict) -> list[tuple[str, int, dict]]:
    """Every round a program names: ('plan', i, spec) and ('rule', j, spec) for
    each rule whose action is a move."""
    out = [("plan", i, s) for i, s in enumerate(prog["plan"])]
    out += [("rule", j, M.ACTIONS[r["do"]][0]) for j, r in enumerate(prog["rules"]) if r["do"] in M.ACTIONS]
    return out


def unsatisfiable(cond: str, prog: dict) -> str | None:
    """Why `cond` can never hold in `prog`, or None. Checked statically: a
    count the opening round is too narrow to reach, a move no rule makes, a
    round kind no round has."""
    if m := M._NUM_RE.fullmatch(cond):
        name, op, k = m.group(1), m.group(2), int(m.group(3))
        if name == "acts":
            return "acts is always equal to step (use step)"
        width = opening_width(prog)
        if name == "r1_majority" and ((op in ("==", ">=") and k > width) or (op == ">" and k >= width)):
            return f"the opening round has only {width} speaker(s)"
        return None
    head, _, tail = cond.partition(":")
    if head in ("last", "ran"):
        if tail not in {r["do"] for r in prog["rules"]}:
            return f"no rule runs {tail}"
    if head in ("last_round", "ran_round"):
        if tail not in {M.round_kind(s) for _, _, s in round_slots(prog)}:
            return f"no round of kind {tail}"
    return None


def opening_width(prog: dict) -> int:
    """The number of speakers of the first round the program runs (r1_majority counts among them):
    the first plan round, or the first round of the move a rule takes before the first round."""
    act = first_action(prog)
    spec = M.ACTIONS[act][0] if act in M.ACTIONS else prog["plan"][0]
    return len(spec["personas"])


def first_action(prog: dict) -> str:
    """The action a program takes before its first round. Nothing has been said
    then, so every condition has a fixed value and this is the same on every
    question (it is exactly what run_program decides at step 0)."""
    st = M.State([], [], [], prog["plan"], 4)
    return next((r["do"] for r in prog["rules"] if all(M._cond(c, st) for c in r["when"])),
                prog["default"])


def validate_v3(prog: dict) -> None:
    if prog["plan"][0].get("sees", D.DEFAULT_SEES) != D.DEFAULT_SEES:
        raise ValueError("the first round has nothing to see: leave out its 'sees' (normalize_program)")
    for rule in prog["rules"]:
        for cond in rule["when"]:
            if (why := unsatisfiable(cond, prog)) is not None:
                raise ValueError(f"condition {cond!r} can never hold: {why}")
    if (act := first_action(prog)).startswith("stop:"):
        raise ValueError(f"the program stops before its first round ({act} at step==0), so it never "
                         "answers: some rule must hold at step==0 and run 'continue' or a round, e.g. "
                         "a first rule {\"when\": [\"plan_left\"], \"do\": \"continue\"}")
    if judge_first(prog, act):
        raise ValueError("a judge cannot speak first: there are no answers to choose between yet")


def judge_first(prog: dict, act: str | None = None) -> bool:
    """Whether the program's first round (run by `act`, its step-0 action) has a judge in it."""
    act = first_action(prog) if act is None else act
    if act.startswith("stop:"):
        return False
    opening = prog["plan"][0] if act == "continue" else M.ACTIONS[act][0]
    return D.JUDGE_PERSONA in opening["personas"]


def reviewer_first(prog: dict) -> bool:
    """Whether the program's first round (the first plan round, or the round a rule runs at step==0)
    has a critic, a verifier or a synthesizer in it (debate_mcq.REVIEWERS). They review the answers
    given so far, and there are none yet, so the cluster search draws no such program and refuses
    such a seed (2026-10-07). Not part of validate_program: archives of earlier runs, where such
    programs were common, still load and replay."""
    act = first_action(prog)
    if act.startswith("stop:"):
        return False
    opening = prog["plan"][0] if act == "continue" else M.ACTIONS[act][0]
    return any(p in D.REVIEWERS for p in opening["personas"])


def canon(prog: dict) -> str:
    return M.canon_prog(prog)


def copy_program(prog: dict) -> dict:
    return json.loads(json.dumps(prog))


# --- the eight protocol seeds ----------------------------------------------------
# Hand-written from the literature, not from any earlier experiment in this
# repo. Each is a recognisable protocol and doubles as a citable baseline.
# Comments give the round-by-round behaviour and the cost in speaker turns.

def CONT() -> dict:
    """A fresh 'run the next plan round' rule (fresh so no two programs share
    a rule object that an in-place edit could corrupt)."""
    return {"when": ["plan_left"], "do": "continue"}


CONTINUE = CONT()          # read-only reference copy

PROTOCOLS: dict[str, dict] = {
    # one solver, read its letter. The one-call floor.
    "direct": {"plan": [solvers(1)], "rules": [CONT()], "default": "stop:last_commit"},
    # four independent solvers, plurality vote. No debate. 4 turns.
    "self_consistency": {"plan": [solvers(4)], "rules": [CONT()], "default": "stop:vote"},
    # Du et al.: three solvers, then two rounds in which each solver sees every
    # prior answer and reconsiders; plurality over all commitments. 9 turns.
    "mad": {"plan": [solvers(3), solvers(3), solvers(3)], "rules": [CONT()],
            "default": "stop:vote"},
    # solver -> critic -> solver -> critic -> solver; read the last. 5 turns.
    "self_refine": {"plan": [solvers(1), PLAN_ROUNDS["critic"], solvers(1),
                             PLAN_ROUNDS["critic"], solvers(1)],
                    "rules": [CONT()], "default": "stop:last_commit"},
    # four solvers, then a verifier tests the committed letters against the
    # question wording; read the verifier. 5 turns.
    "verify_then_decide": {"plan": [solvers(4)],
                           "rules": [CONT(), {"when": ["step==1"], "do": "verifier"}],
                           "default": "stop:last_commit"},
    # ARMOR-MAD style: four solvers; if three or more agree, stop and vote;
    # otherwise two critic rounds, then vote. 4 or 6 turns.
    "early_exit_agree": {"plan": [solvers(4)],
                         "rules": [CONT(),
                                   {"when": ["step==1", "r1_majority>=3"], "do": "stop:vote"},
                                   {"when": ["step==1"], "do": "critic"},
                                   {"when": ["step==2"], "do": "critic"}],
                         "default": "stop:vote"},
    # two solvers; if they disagree, two blind independent solvers join; vote.
    # 2 or 4 turns.
    "fresh_on_disagree": {"plan": [solvers(2)],
                          "rules": [CONT(),
                                    {"when": ["step==1", "n_distinct>=2"], "do": "fresh"}],
                          "default": "stop:vote"},
    # a blind expert first, then an expert+solver pair that sees only the
    # expert; stop if the pair agrees, otherwise a verifier closes. 3 or 4 turns.
    "expert_first": {"plan": [PLAN_ROUNDS["expert_blind"]],
                     "rules": [CONT(),
                               {"when": ["step==1"], "do": "pair_expert_solver"},
                               {"when": ["step==2", "last_round_agree"], "do": "stop:last_commit"},
                               {"when": ["step==2"], "do": "verifier"}],
                     "default": "stop:last_commit"},
}


def protocols_v3() -> dict[str, dict]:
    """The literature programs in the v3 grammar, in their fixed order (the
    seed stage takes a prefix when it has fewer places). The first eight are the
    v2 programs, written in v3 names: 'fresh' is two blind solvers, and the
    expert+solver pair sees everything (after a lone expert that is the same as
    seeing the last round). Then high-effort variants, each within the 16-turn
    cap; the last is also a blind variant (a blind high-effort solver settles a
    disagreement)."""
    V = v3_vocabulary()

    def r(name: str) -> dict:
        return dict(V[name])

    def prog(plan: list[str], rules: list[dict], default: str) -> dict:
        return {"plan": [r(x) for x in plan], "rules": [CONT()] + rules, "default": default}

    return {
        "direct": prog(["solver"], [], "stop:last_commit"),
        "self_consistency": prog(["solver_x4"], [], "stop:vote"),
        "mad": prog(["solver_x3", "solver_x3", "solver_x3"], [], "stop:vote"),
        "self_refine": prog(["solver", "critic", "solver", "critic", "solver"], [], "stop:last_commit"),
        "verify_then_decide": prog(["solver_x4"], [{"when": ["step==1"], "do": "verifier"}], "stop:last_commit"),
        "early_exit_agree": prog(["solver_x4"], [{"when": ["step==1", "r1_majority>=3"], "do": "stop:vote"},
                                                 {"when": ["step==1"], "do": "critic"},
                                                 {"when": ["step==2"], "do": "critic"}], "stop:vote"),
        "fresh_on_disagree": prog(["solver_x2"], [{"when": ["step==1", "n_distinct>=2"], "do": "solver_x2|blind"}],
                                  "stop:vote"),
        "expert_first": prog(["expert"], [{"when": ["step==1"], "do": "expert_solver"},
                                          {"when": ["step==2", "last_round_agree"], "do": "stop:last_commit"},
                                          {"when": ["step==2"], "do": "verifier"}], "stop:last_commit"),
        # one high-effort solver: the "just think longer" program (5 turns)
        "direct_high": prog(["solver|high"], [], "stop:last_commit"),
        # Self-Refine at high effort, as the external baseline runs it (baselines/selfrefine.py, 2 rounds):
        # answer, then up to 2 feedback -> refine rounds (a critic, then a solver), stopping after a
        # critic that keeps the answer it reviewed, as the external one stops when its feedback says
        # "it is correct" (on Qwen3.5-9B: after the first feedback in 76% of runs). It gets 2 rounds when
        # 5 high-effort speakers fit the turn cap (15 turns at 3 a speaker), else 1 (15 turns at 5 a
        # speaker). Until 2026-10-06 it was solver > critic > solver with no stop.
        "self_refine_high": {"plan": [r(x) for x in ["solver|high"] + ["critic|high", "solver|high"]
                                      * (2 if 5 * D.HIGH_COST <= MAX_TURNS else 1)],
                             "rules": [{"when": ["last_round:critic", "kept_answer"], "do": "stop:last_commit"},
                                       CONT()],
                             "default": "stop:last_commit"},
        # three high-effort solvers, plurality vote (15 turns)
        "self_consistency_high": prog(["solver_x3|high"], [], "stop:vote"),
        # four solvers, then a high-effort verifier (9 turns)
        "verify_then_decide_high": prog(["solver_x4"], [{"when": ["step==1"], "do": "verifier|high"}],
                                        "stop:last_commit"),
        # two solvers; if they disagree, a blind high-effort solver decides (2 or 7 turns)
        "fresh_on_disagree_high": prog(["solver_x2"], [{"when": ["step==1", "n_distinct>=2"], "do": "solver|high|blind"},
                                                       {"when": ["step==2"], "do": "stop:last_commit"}], "stop:vote"),
    }


def judge_seeds() -> dict[str, dict]:
    """The seed programs built on the judge (--judge-persona only; see program_seeds_cluster).
    Both end with a high-effort judge, and only when the answers so far disagree, so a
    unanimous debate costs nothing extra; the final read is the judge's pick (or the
    unanimous answer). Costs at 3 turns a high-effort speaker: 9 or 12, and 8 or 11."""
    if not D.JUDGE_ON:
        return {}
    V = v3_vocabulary()

    def prog(plan: list[str], rules: list[dict]) -> dict:
        return {"plan": [dict(V[x]) for x in plan], "rules": [CONT()] + rules, "default": "stop:last_commit"}

    return {
        # three high-effort solvers; if they disagree, a high-effort judge picks among them
        "judge_on_disagree_high": prog(["solver_x3|high"],
                                       [{"when": ["step==1", "n_distinct>=2"], "do": "judge|high"}]),
        # two high-effort and two blind low-effort solvers (four independent answers);
        # if they disagree, a high-effort judge picks
        "pool_judge_high": prog(["solver_x2|high", "solver_x2|blind"],
                                [{"when": ["step==2", "n_distinct>=2"], "do": "judge|high"}]),
    }


# --- random programs and mutation ----------------------------------------------

def random_plan(rng: random.Random) -> list[dict]:
    if D.V3:
        return random_plan_v3(rng)
    plan = [solvers(rng.choice(OPENING_WIDTHS))]
    while len(plan) < MAX_PLAN and rng.random() < 0.35:
        plan.append(copy_program(rng.choice(list(PLAN_ROUNDS.values()))))
    return plan


def random_plan_v3(rng: random.Random) -> list[dict]:
    """An opening round of 1 to 4 solvers at either effort, then, with
    probability 0.35 each, more rounds drawn from the whole vocabulary."""
    openings = [s for name, s in PLAN_ROUNDS.items() if name.split("|")[0].startswith("solver") and "sees" not in s]
    plan = [copy_program(rng.choice(openings))]
    while len(plan) < MAX_PLAN and rng.random() < 0.35:
        plan.append(copy_program(rng.choice(list(PLAN_ROUNDS.values()))))
    return normalize_program({"plan": plan, "rules": [], "default": "stop:vote"})["plan"]


def rand_cond(rng: random.Random, prog: dict | None = None) -> str:
    """A random condition: evolve_program_mcq's under v2; under v3 one that can
    hold in `prog` (random_condition_v3)."""
    return random_condition_v3(rng, prog) if D.V3 else M.random_condition(rng)


def random_condition_v3(rng: random.Random, prog: dict) -> str:
    """One of five families, drawn uniformly (only four if the program makes no
    move), with values that can hold in `prog`: the moves its rules make, the
    round kinds it has, the width of its opening round. No `acts`: it always
    equals `step`."""
    moves = sorted({r["do"] for r in prog["rules"] if not r["do"].startswith("stop:")})
    kinds = sorted({M.round_kind(s) for _, _, s in round_slots(prog)})
    width = opening_width(prog)
    fam = rng.choice(["step", "flag", "count", "kind"] + (["move"] if moves else []))
    if fam == "step":
        return f"step{rng.choice(['==', '>=', '<'])}{rng.randrange(7)}"
    if fam == "flag":
        return rng.choice(M._BOOL_CONDS)
    if fam == "count":
        if rng.random() < 0.5:
            return f"r1_majority{rng.choice(['==', '>=', '<'])}{rng.randint(1, width)}"
        return f"n_distinct{rng.choice(['==', '>='])}{rng.randrange(1, 5)}"
    if fam == "move":
        return f"{rng.choice(['last', 'ran'])}:{rng.choice(moves)}"
    return f"{rng.choice(['last_round', 'ran_round'])}:{rng.choice(kinds)}"


def rand_act(rng: random.Random) -> str:
    """A random action. Under v3, drawn level by level, each level uniform:
    continue / a stop / a move; then the stop's read, or the move's speakers,
    effort and visibility (a combination outside the grammar is drawn again)."""
    if not D.V3:
        return M.random_action(rng)
    kind = rng.choice(["continue", "stop", "move"])
    if kind == "continue":
        return "continue"
    if kind == "stop":
        return "stop:" + rng.choice(M.STOP_READS)
    while True:
        base = rng.choice(list(speakers_v3()))
        name = v3_name(v3_spec(base, rng.random() < 0.5, base in BLINDABLE and rng.random() < 0.5))
        if name in M.ACTIONS:
            return name


def random_program(rng: random.Random, min_rules: int = 3, max_rules: int = 8) -> dict:
    """A program drawn from the grammar. Most such programs are useless in a
    predictable way (stop at once, or run to the cap); the seed stage's sanity
    run filters those. A plan_left->continue rule is included most of the time,
    otherwise the opening rounds would rarely run at all."""
    if D.V3:
        return random_program_v3(rng, min_rules, max_rules)
    rules = [{"when": [M.random_condition(rng) for _ in range(rng.choice([1, 1, 2]))],
              "do": M.random_action(rng)}
             for _ in range(rng.randint(min_rules, max_rules))]
    if rng.random() < 0.7:
        rules.insert(rng.randrange(len(rules) + 1), CONT())
    prog = {"plan": random_plan(rng), "rules": rules,
            "default": "stop:" + rng.choice(M.STOP_READS)}
    validate_program(prog)
    return prog


def random_program_v3(rng: random.Random, min_rules: int = 3, max_rules: int = 8) -> dict:
    """As random_program, but the actions are drawn first and the conditions
    after, so every condition can hold in the finished program. A draw that
    would stop before its first round (invalid under v3), or open with a judge,
    is drawn again; so is one whose conditions turn out unable to hold once all are
    drawn (an r1_majority drawn for one opening round, when a later rule makes
    another round run first)."""
    for _ in range(1000):
        rules = [{"when": [], "do": rand_act(rng)} for _ in range(rng.randint(min_rules, max_rules))]
        if rng.random() < 0.7:
            rules.insert(rng.randrange(len(rules) + 1), CONT())
        prog = {"plan": random_plan(rng), "rules": rules, "default": "stop:" + rng.choice(M.STOP_READS)}
        for rule in rules:
            if not rule["when"]:
                rule["when"] = [random_condition_v3(rng, prog) for _ in range(rng.choice([1, 1, 2]))]
        if not (act := first_action(prog)).startswith("stop:") and not judge_first(prog, act):
            try:
                validate_program(prog)
            except ValueError:
                continue
            return prog
    raise RuntimeError("no random program that starts a debate in 1000 draws")


PLAN_OPS = ("plan_width", "plan_replace", "plan_add", "plan_drop")


def mutate_plan(prog: dict, rng: random.Random, op: str | None = None) -> tuple[dict, str]:
    """One edit to the opening rounds. Returns (child, op name). With `op`
    None the kind is drawn here (the v2 search); the cluster search names it. Under
    v3 a width change keeps the round's effort."""
    p = copy_program(prog)
    plan = p["plan"]
    if op is None:
        ops = ["plan_width", "plan_replace"]
        if len(plan) < MAX_PLAN:
            ops.append("plan_add")
        if len(plan) > 1:
            ops.append("plan_drop")
        op = rng.choice(ops)
    if op == "plan_width":
        first = plan[0]
        if first["personas"] and all(x == "solver" for x in first["personas"]) \
                and "sees" not in first:
            widths = [w for w in OPENING_WIDTHS if w != len(first["personas"])]
            plan[0] = solvers(rng.choice(widths))
        else:
            plan[0] = solvers(rng.choice(OPENING_WIDTHS))
        if first.get("effort", D.DEFAULT_EFFORT) != D.DEFAULT_EFFORT:
            plan[0]["effort"] = first["effort"]
    elif op == "plan_replace":
        i = rng.randrange(len(plan))
        choices = [s for s in PLAN_ROUNDS.values() if s != plan[i]]
        plan[i] = copy_program(rng.choice(choices))
    elif op == "plan_add":
        plan.insert(rng.randrange(len(plan) + 1), copy_program(rng.choice(list(PLAN_ROUNDS.values()))))
    else:
        plan.pop(rng.randrange(len(plan)))
    if D.V3:
        p = normalize_program(p)
    return p, op


RULE_OPS = ("change_action", "add_rule", "change_cond", "change_default", "drop_rule", "swap_rules")
ALL_OPS = RULE_OPS + ("plan",)


def _pick(n: int, rng: random.Random, weights: list[float] | None) -> int:
    """A rule index: uniform, or in proportion to `weights` when given."""
    if weights is None or len(weights) != n:
        return rng.randrange(n)
    return rng.choices(range(n), weights=weights)[0]


def mutate_rules(prog: dict, rng: random.Random, op: str,
                 weights: list[float] | None = None) -> tuple[dict, str]:
    """One rule edit of the named kind (evolve_program_mcq's operators, but with
    the kind chosen by the caller so the guided mutator and the logs can name
    it). Falls back to the closest legal kind when the named one cannot apply;
    returns (child, the kind actually applied). With `weights` (one per rule),
    the rule an edit touches is picked in proportion to them."""
    p = copy_program(prog)
    rules = p["rules"]
    if op in ("drop_rule", "swap_rules") and len(rules) < 2:
        op = "change_action"
    if op == "add_rule" and len(rules) >= M.MAX_RULES:
        op = "change_cond"
    if op in ("change_action", "change_cond") and not rules:
        op = "add_rule"
    if op == "change_action":
        rules[_pick(len(rules), rng, weights)]["do"] = rand_act(rng)
    elif op == "add_rule":
        rule = {"when": [rand_cond(rng, prog) for _ in range(rng.choice([1, 1, 2]))],
                "do": rand_act(rng)}
        rules.insert(rng.randrange(len(rules) + 1), rule)
    elif op == "change_cond":
        rule = rules[_pick(len(rules), rng, weights)]
        if rule["when"] and rng.random() < 0.7:
            rule["when"][rng.randrange(len(rule["when"]))] = rand_cond(rng, prog)
        else:
            rule["when"] = rule["when"] + [rand_cond(rng, prog)]
    elif op == "change_default":
        p["default"] = "stop:" + rng.choice([r for r in M.STOP_READS
                                              if "stop:" + r != p["default"]])
    elif op == "drop_rule":
        rules.pop(_pick(len(rules), rng, weights))
    elif op == "swap_rules":
        if weights is None or len(weights) != len(rules):
            i, j = rng.sample(range(len(rules)), 2)
        else:
            i = _pick(len(rules), rng, weights)
            rest = [k for k in range(len(rules)) if k != i]
            j = rest[_pick(len(rest), rng, [weights[k] for k in rest])]
        rules[i], rules[j] = rules[j], rules[i]
    else:
        raise ValueError(op)
    return p, op


def draw_op(rng: random.Random, plan_prob: float = 0.2) -> str:
    """The kind of edit to make, drawn before anyone (random or guided) makes it."""
    if rng.random() < plan_prob:
        return "plan"
    return rng.choice(RULE_OPS)


def mutate(prog: dict, rng: random.Random, op: str | None = None) -> tuple[dict, str]:
    """One random edit. Returns (child, op name). The child is validated; an
    invalid child is retried with a fresh draw (a handful of times at most)."""
    for _ in range(20):
        o = op or draw_op(rng)
        if o == "plan":
            child, sub = mutate_plan(prog, rng)
            name = sub
        else:
            child, name = mutate_rules(prog, rng, o)
        try:
            validate_program(child)
        except (AssertionError, ValueError):
            continue
        if canon(child) != canon(prog):
            return child, name
    return copy_program(prog), "none"


# --- round edits and crossover (v3) -------------------------------------------------------
# set_effort / set_visibility switch one round of the program, a plan round or the
# move a rule makes, between low and high effort / between seeing and blind.
# crossover joins one program's opening rounds to another program's rules and default.

ROUND_OPS = ("set_effort", "set_visibility")
CROSSOVER = "crossover"


def _toggled(spec: dict, what: str) -> dict:
    s = norm_spec(spec)
    if what == "effort":
        if s.pop("effort", None) is None:
            s["effort"] = "high"
    elif s.pop("sees", None) is None:
        s["sees"] = "none"
    return s


def round_edit_slots(prog: dict, what: str) -> list[tuple[str, int, dict]]:
    """The rounds whose effort ('effort') or visibility ('visibility') can be
    switched with the result still in the grammar, each with its switched spec.
    The first plan round has no visibility to switch."""
    out = []
    for where, i, spec in round_slots(prog):
        if what == "visibility" and where == "plan" and i == 0:
            continue
        new = _toggled(spec, what)
        if new in PLAN_ROUNDS.values():
            out.append((where, i, new))
    return out


def mutate_round(prog: dict, rng: random.Random, op: str) -> tuple[dict, str]:
    p = copy_program(prog)
    slots = round_edit_slots(prog, "effort" if op == "set_effort" else "visibility")
    if slots:
        where, i, new = rng.choice(slots)
        if where == "plan":
            p["plan"][i] = new
        else:
            p["rules"][i]["do"] = v3_name(new)
    return normalize_program(p), op


def crossover(prog: dict, donor: dict, rng: random.Random) -> dict:
    """`prog`'s opening rounds with `donor`'s rules and default, or the other
    way round (even odds)."""
    if rng.random() < 0.5:
        child = {"plan": prog["plan"], "rules": donor["rules"], "default": donor["default"]}
    else:
        child = {"plan": donor["plan"], "rules": prog["rules"], "default": prog["default"]}
    return normalize_program(copy_program(child))


# The global search draws one kind uniformly from the kinds that can apply to the program at hand
# (mutate_uniform); the cluster search draws a family first (mutate_by_family, since 2026-10-07).
EDIT_KINDS = RULE_OPS + PLAN_OPS + ROUND_OPS + (CROSSOVER,)

# --any-round-width: set_width takes plan_width's place (so the number of kinds is the same). It
# changes the number of solvers of ONE solver round, chosen uniformly among the program's solver
# rounds (the first plan round, a later one, or an extra round run by a rule), to another width
# drawn uniformly from those that exist at the round's effort and visibility.
ANY_ROUND_WIDTH = False
WIDTH_OP = "set_width"


def edit_kinds() -> tuple[str, ...]:
    """The edit kinds of this run: EDIT_KINDS, with set_width for plan_width under --any-round-width."""
    if not ANY_ROUND_WIDTH:
        return EDIT_KINDS
    return tuple(WIDTH_OP if k == "plan_width" else k for k in EDIT_KINDS)


def width_edit_slots(prog: dict) -> list[tuple[str, int, dict]]:
    """(where, index, new spec) for every solver round of `prog` and every other width it can
    take with its effort and visibility kept (the result must be a round of the vocabulary)."""
    out = []
    for where, i, spec in round_slots(prog):
        s = norm_spec(spec)
        if set(s["personas"]) != {"solver"}:
            continue
        for w in OPENING_WIDTHS:
            new = norm_spec({**s, "personas": ["solver"] * w})
            if w != len(s["personas"]) and new in PLAN_ROUNDS.values():
                out.append((where, i, new))
    return out


def mutate_width(prog: dict, rng: random.Random) -> tuple[dict, str]:
    """set_width: one solver round (uniform over the program's solver rounds) to another width."""
    p = copy_program(prog)
    opts = width_edit_slots(prog)
    if opts:
        where, i = rng.choice(sorted({(w, j) for w, j, _ in opts}))
        new = rng.choice([s for w, j, s in opts if (w, j) == (where, i)])
        if where == "plan":
            p["plan"][i] = new
        else:
            p["rules"][i]["do"] = v3_name(new)
    return normalize_program(p), WIDTH_OP


def applicable_edits(prog: dict, donors: list[dict] | tuple = ()) -> list[str]:
    """The edit kinds that can apply to `prog`, in edit_kinds() order. The round
    edits need the v3 grammar; crossover needs another program to take from; the
    width edit needs a solver round with another width in the vocabulary."""
    n_rules, n_plan = len(prog["rules"]), len(prog["plan"])
    out = []
    for op in edit_kinds():
        if op in ("drop_rule", "swap_rules") and n_rules < 2:
            continue
        if op in ("change_action", "change_cond") and n_rules < 1:
            continue
        if op == "add_rule" and n_rules >= M.MAX_RULES:
            continue
        if op == "plan_add" and n_plan >= MAX_PLAN:
            continue
        if op == "plan_drop" and n_plan <= 1:
            continue
        if op in ROUND_OPS and not (D.V3 and round_edit_slots(
                prog, "effort" if op == "set_effort" else "visibility")):
            continue
        if op == CROSSOVER and not donors:
            continue
        if op == WIDTH_OP and not width_edit_slots(prog):
            continue
        out.append(op)
    return out


def mutate_uniform(prog: dict, rng: random.Random, weights: list[float] | None = None,
                   donors: list[dict] | tuple = ()) -> tuple[dict, str]:
    """One random edit whose kind is drawn uniformly from the applicable kinds.
    Returns (child, op name). An invalid or unchanged child is redrawn within
    the same kind, so kinds whose edits are more often invalid are not drawn
    less; a kind that yields nothing in 20 tries is set aside and another kind
    drawn. `weights` (one per rule) steer which rule a rule edit touches;
    `donors` are the programs a crossover may take from."""
    kinds = applicable_edits(prog, donors)
    while kinds:
        op = rng.choice(kinds)
        if (got := _edit_of_kind(prog, rng, op, weights, donors)) is not None:
            return got
        kinds.remove(op)
    return copy_program(prog), "none"


def _edit_of_kind(prog: dict, rng: random.Random, op: str, weights: list[float] | None,
                  donors: list[dict] | tuple) -> tuple[dict, str] | None:
    """One valid, changed child of `prog` by an edit of kind `op`, in at most 20 tries; None if
    none of them gives one."""
    for _ in range(20):
        if op in PLAN_OPS:
            child, name = mutate_plan(prog, rng, op)
        elif op in ROUND_OPS:
            child, name = mutate_round(prog, rng, op)
        elif op == WIDTH_OP:
            child, name = mutate_width(prog, rng)
        elif op == CROSSOVER:
            child, name = crossover(prog, rng.choice(list(donors)), rng), op
        else:
            child, name = mutate_rules(prog, rng, op, weights)
        try:
            validate_program(child)
        except (AssertionError, ValueError):
            continue
        if canon(child) != canon(prog):
            return child, name
    return None


# The cluster search's draw (2026-10-07): a family first, evenly among the families with a kind that
# applies, then a kind evenly within it. Over five earlier runs (drawn evenly over the kinds), 7% of
# the children of rule edits beat their parent, against 12% for the other edits; with six rule kinds
# among thirteen, rule edits made 37% of the children (16% by family). The global search keeps the
# draw over kinds (mutate_uniform).
EDIT_FAMILIES: dict[str, tuple[str, ...]] = {
    "rule": RULE_OPS, "plan": ("plan_replace", "plan_add", "plan_drop"), "width": ("plan_width", WIDTH_OP),
    "effort": ("set_effort",), "visibility": ("set_visibility",), "crossover": (CROSSOVER,)}


def edit_family(op: str) -> str:
    return next(f for f, ops in EDIT_FAMILIES.items() if op in ops)


def mutate_by_family(prog: dict, rng: random.Random, weights: list[float] | None = None,
                     donors: list[dict] | tuple = ()) -> tuple[dict, str]:
    """One random edit: a family drawn evenly from the families with an applicable kind (rule,
    plan, width, effort, visibility, crossover; EDIT_FAMILIES), then a kind drawn evenly from that
    family's applicable kinds. As in mutate_uniform, an invalid or unchanged child is redrawn within
    the kind, and a kind that yields nothing in 20 tries is set aside (with its family once the
    family has none left). Returns (child, op name)."""
    families: dict[str, list[str]] = {}
    for op in applicable_edits(prog, donors):
        families.setdefault(edit_family(op), []).append(op)
    while families:
        fam = rng.choice(list(families))
        op = rng.choice(families[fam])
        if (got := _edit_of_kind(prog, rng, op, weights, donors)) is not None:
            return got
        families[fam].remove(op)
        if not families[fam]:
            del families[fam]
    return copy_program(prog), "none"


def prune(prog: dict, fires: dict[int, int]) -> tuple[dict, list[int]]:
    """`prog` without the rules that never decided a step, and the kept rules'
    weights for choosing which rule an edit touches (times it decided a step,
    plus one). `fires` maps rule index -> times it was the matching rule over
    the recorded questions. A rule that never matched first changes nothing on
    those questions, so the pruned program behaves the same on them. The whole
    program is kept if nothing would be left or the result is not valid."""
    keep = [i for i in range(len(prog["rules"])) if fires.get(i, 0) > 0]
    if keep and len(keep) < len(prog["rules"]):
        cand = copy_program(prog)
        cand["rules"] = [cand["rules"][i] for i in keep]
        try:
            validate_program(cand)
            return cand, [fires[i] + 1 for i in keep]
        except (AssertionError, ValueError):
            pass
    return copy_program(prog), [fires.get(i, 0) + 1 for i in range(len(prog["rules"]))]


def op_family(op: str) -> str:
    return "plan" if op.startswith("plan") else op


def expected_rule_delta(op: str) -> int | None:
    """How the rule count must change under an edit of this kind (None = any)."""
    return {"drop_rule": -1, "add_rule": 1}.get(op_family(op), 0 if op != "none" else None)


# --- structural distance -----------------------------------------------------------

def cond_family(cond: str) -> str:
    if m := M._NUM_RE.fullmatch(cond):
        return m.group(1)
    head, _, tail = cond.partition(":")
    if head in ("last", "ran", "last_round", "ran_round"):
        return f"{head}:{tail}"
    return cond


def struct_tokens(prog: dict) -> frozenset[str]:
    """What a program is made of, as a set: its opening rounds, the condition
    families and actions its rules use, and how it stops."""
    toks = set()
    for i, spec in enumerate(prog["plan"]):
        toks.add(f"plan{i}:{plan_round_name(spec)}")
    toks.add(f"open_width:{len(prog['plan'][0]['personas'])}")
    toks.add(f"plan_len:{len(prog['plan'])}")
    for rule in prog["rules"]:
        toks.add(f"act:{rule['do']}")
        for c in rule["when"]:
            toks.add(f"cond:{cond_family(c)}")
    toks.add(f"default:{prog['default']}")
    n = len(prog["rules"])
    toks.add("nrules:" + ("1-3" if n <= 3 else "4-7" if n <= 7 else "8+"))
    return frozenset(toks)


def struct_distance(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return 1.0 - len(a & b) / union if union else 0.0


def farthest_point(candidates: list, n_pick: int, dist, fixed: list | None = None) -> list[int]:
    """Indices into `candidates` of the n_pick items farthest (max-min) from the
    `fixed` items and from each other. `dist(x, y)` is any distance."""
    chosen: list = list(fixed or [])
    picked: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(picked) < n_pick:
        if chosen:
            best = max(remaining, key=lambda i: min(dist(candidates[i], c) for c in chosen))
        else:
            best = remaining[0]
        picked.append(best)
        chosen.append(candidates[best])
        remaining.remove(best)
    return picked


# --- per-question results ----------------------------------------------------------
# A program's record on one replicate is a map question id -> [mark, turns,
# letter, path]. Marks are 0/1; turns are speaker turns (the program-intrinsic
# cost; under v2 every answering turn is two model calls, reasoning plus
# summary, so real calls are about double); path is the action sequence.

def path_of(out: dict) -> str:
    return ",".join(out["actions"])


def cap_cut(program: dict, entry: list) -> bool:
    """Whether the per-question turn cap stopped this recorded debate (an entry of
    ProgRecord.record): its last decision was a rule's round (not a stop, not a continue past the
    plan), yet the debate ended there. It reads only the recorded path and rules fired, so it works
    on every archive with rules fired on record (evolve_program_mcq.run_program's "capped" says the
    same as it runs)."""
    if len(entry) < 5 or not entry[4]:
        return False
    acts = [a for a in (entry[3] or "").split(",") if a]
    fired = [int(x) for x in str(entry[4]).split(",") if x != ""]
    if len(fired) == len(acts):                           # it ran out of decisions before any stop: cut
        return True                                       # (run_program's step limit before 2026-10-06)
    if len(fired) != len(acts) + 1 or fired[-1] < 0:      # it stopped by the default
        return False
    do = program["rules"][fired[-1]]["do"]
    if do.startswith("stop:") or (do == "continue" and acts.count("continue") >= len(program["plan"])):
        return False
    return True


def cap_cuts(rec: "ProgRecord", qids: list[str], rep: int = 0) -> int:
    """How many of `qids` the turn cap stopped at replicate `rep` (questions not run are not counted)."""
    got = rec.reps.get(rep, {})
    return sum(cap_cut(rec.program, got[q]) for q in qids if q in got)


def behaviour_hash(rec: dict[str, list]) -> str:
    """Identity of a fully covered record: what it did, what it answered and how many turns it
    used on every question. Two programs with the same hash are the same individual. The turns
    are part of it (since 2026-10-06): a program that answers like another at fewer turns is a
    cheaper program, not a duplicate (blind speakers are shared, so such twins are common)."""
    blob = json.dumps(sorted((q, v[1], v[2], v[3]) for q, v in rec.items()), separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class ProgRecord:
    """One scored program: its text, provenance and per-question results."""

    def __init__(self, program: dict, name: str, lineage: str, gen: int,
                 parent: str | None = None, op: str | None = None, guided: bool = False):
        self.program = program
        self.key = canon(program)
        self.name, self.lineage, self.gen = name, lineage, gen
        self.parent, self.op, self.guided = parent, op, guided
        self.reps: dict[int, dict[str, list]] = {}
        self.tokens = struct_tokens(program)
        self.live_turns = 0
        # key of an earlier program with the same behaviour on every search
        # question, if any: a duplicate individual, kept on file but never
        # selected. Recomputed by the search whenever the question set changes.
        self.dup_of: str | None = None
        # free-form provenance a search may attach (the cluster search records the
        # slot and target group a child was bred for)
        self.meta: dict = {}

    # -- serialisation --
    def to_json(self) -> dict:
        d = {"key": self.key, "program": self.program, "name": self.name,
             "lineage": self.lineage, "gen": self.gen, "parent": self.parent,
             "op": self.op, "guided": self.guided, "live_turns": self.live_turns,
             "dup_of": self.dup_of,
             "reps": {str(r): rec for r, rec in self.reps.items()}}
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_json(cls, d: dict) -> "ProgRecord":
        p = cls(d["program"], d["name"], d["lineage"], d["gen"], d.get("parent"),
                d.get("op"), bool(d.get("guided")))
        p.live_turns = d.get("live_turns", 0)
        p.dup_of = d.get("dup_of")
        p.meta = dict(d.get("meta") or {})
        p.reps = {int(r): {q: list(v) for q, v in rec.items()} for r, rec in d["reps"].items()}
        return p

    # -- results --
    def record(self, rep: int, qid: str, out: dict) -> None:
        """[mark, turns, letter, path, rules fired, completion tokens]. The last
        two are new: which rule decided each step ('-1' = none, the default), and
        the tokens the debate's recordings used (None if not logged)."""
        self.reps.setdefault(rep, {})[qid] = [int(bool(out["correct"])), out["n_calls"],
                                              out["letter"] or "?", path_of(out),
                                              ",".join(str(i) for i in out.get("fired", [])),
                                              out.get("tokens")]

    def rule_fires(self) -> Counter:
        """Rule index -> how many steps it decided, over every recorded
        question and replicate."""
        c: Counter = Counter()
        for rec in self.reps.values():
            for v in rec.values():
                if len(v) > 4 and v[4]:
                    c.update(int(i) for i in v[4].split(",") if i != "-1")
        return c

    def tokens_used(self, qids: list[str], rep: int = 0) -> float | None:
        """Mean completion tokens per covered question at `rep`; None if none logged."""
        rec = self.reps.get(rep, {})
        vals = [rec[q][5] for q in qids if q in rec and len(rec[q]) > 5 and rec[q][5] is not None]
        return sum(vals) / len(vals) if vals else None

    def has(self, rep: int, qid: str) -> bool:
        return qid in self.reps.get(rep, {})

    def covered(self, qids: list[str], rep: int = 0) -> int:
        rec = self.reps.get(rep, {})
        return sum(1 for q in qids if q in rec)

    def gaps(self, qids: list[str], rep: int = 0) -> list[str]:
        rec = self.reps.get(rep, {})
        return [q for q in qids if q not in rec]

    def mark(self, qid: str) -> float | None:
        """Mean mark over the replicates that have this question; None if none."""
        vals = [rec[qid][0] for rec in self.reps.values() if qid in rec]
        return sum(vals) / len(vals) if vals else None

    def score(self, qids: list[str]) -> float:
        """Mean mark over `qids`; a question with no result counts as wrong."""
        if not qids:
            return 0.0
        return sum(self.mark(q) or 0.0 for q in qids) / len(qids)

    def n_correct(self, qids: list[str]) -> float:
        return sum(self.mark(q) or 0.0 for q in qids)

    def turns(self, qids: list[str], rep: int = 0) -> float | None:
        """Mean speaker turns per covered question at `rep`; None if uncovered."""
        rec = self.reps.get(rep, {})
        vals = [rec[q][1] for q in qids if q in rec]
        return sum(vals) / len(vals) if vals else None

    def behaviour(self, qids: list[str]) -> str | None:
        rec = self.reps.get(0, {})
        if any(q not in rec for q in qids):
            return None
        return behaviour_hash({q: rec[q] for q in qids})


def semantic_distance(a: ProgRecord, b: ProgRecord, qids: list[str]) -> float:
    """How differently two programs behaved at replicate 0: the fraction of
    questions with a different action path, averaged with the fraction with a
    different final letter. Questions either lacks count as different."""
    ra, rb = a.reps.get(0, {}), b.reps.get(0, {})
    if not qids:
        return 0.0
    diff_path = diff_letter = 0
    for q in qids:
        x, y = ra.get(q), rb.get(q)
        if x is None or y is None:
            diff_path += 1
            diff_letter += 1
            continue
        diff_path += x[3] != y[3]
        diff_letter += x[2] != y[2]
    return (diff_path + diff_letter) / (2 * len(qids))


def combined_distance(a: ProgRecord, b: ProgRecord, qids: list[str]) -> float:
    return 0.5 * (struct_distance(a.tokens, b.tokens) + semantic_distance(a, b, qids))


# --- the live runner ---------------------------------------------------------------------

def model_tag(model: str) -> str:
    return model.rsplit("/", 1)[-1].replace(".", "").replace("-", "_").lower()


def make_runner(rows: dict, cache_path: Path, base_urls: str = DEFAULT_BASE_URLS,
                model: str = DEFAULT_MODEL, temperature: float = 0.7,
                max_total_calls: int | None = None, api_key: str = "EMPTY",
                lock: bool = True, progress: bool = True):
    """The budgeted live runner over ONE fresh cache file. No cache of any
    earlier experiment is loaded: the executor settings differ from theirs, so
    their recordings could not be replayed anyway, and the key does not name
    the model, so a stale file could only mislead. Budget starts at 0 (cache
    only); callers open it with reset_budget()."""
    return M.BudgetedRunner(rows, [], lock=lock, base_urls=base_urls, model=model,
                            temperature=temperature, answer_tokens=ANSWER_TOKENS,
                            cache_path=Path(cache_path), max_calls=max_total_calls,
                            api_key=api_key, progress=progress)


# --- running a program over questions -------------------------------------------------

def run_many(prog: dict, runner, rows: dict, qids: list[str], rep: int, max_calls: int,
             workers: int = 1, desc: str = "") -> dict[str, dict | None]:
    """`prog` on each of `qids` at `rep`. A question the runner would not
    serve (off-cache under the current budget) maps to None. Questions run in
    parallel threads when workers > 1; the runners are thread-safe."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def one(q: str):
        try:
            return q, M.run_program(prog, runner, rows[q], rep=rep, max_calls=max_calls)
        except M.OffCache:
            return q, None

    if workers > 1 and len(qids) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(pool.map(one, qids), total=len(qids), unit="q", desc=desc,
                                leave=False, disable=len(qids) < 40))
    else:
        results = [one(q) for q in tqdm(qids, unit="q", desc=desc, leave=False,
                                        disable=len(qids) < 40)]
    return dict(results)


def run_pairs(items: list[tuple], runner, rows: dict, rep: int, max_calls: int,
              workers: int = 1, desc: str = "") -> dict[tuple, dict | None]:
    """Many (tag, program, question) triples in one thread pool. Returns
    {(tag, question): result or None}."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def one(item):
        tag, prog, q = item
        try:
            return (tag, q), M.run_program(prog, runner, rows[q], rep=rep, max_calls=max_calls)
        except M.OffCache:
            return (tag, q), None

    if workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(pool.map(one, items), total=len(items), unit="q", desc=desc,
                                leave=False, disable=len(items) < 40))
    else:
        results = [one(it) for it in tqdm(items, unit="q", desc=desc, leave=False,
                                          disable=len(items) < 40)]
    return dict(results)


# --- question groups -----------------------------------------------------------------

def load_groups(path: Path, per_group: int, dev_split: Path | None = None) -> dict:
    """The search questions and held-out questions per group. `subset` lists
    in the clusters file are ordered (centre, then alternating farthest and
    random picks) and the order does not depend on how many were requested,
    so the first `per_group` of each is the search set at that size and any
    later question of the group is held out. `per_group` is a cap: a group
    with fewer questions is used whole. `per_group` 0 makes every question of
    every group a search question. The groups' challenge profile is read from
    `challenges` (clusters files since describer prompt v3) or `risks`
    (earlier ones).

    `dev_split` (off by default): a split file of split_train_dev.py. A group's questions in
    its dev list are then held out (for --pick-champions) and the rest are search questions,
    each in the clusters file's order; every question of every group must be in the split.
    With `per_group` N > 0 as well (HLE, 2026-10-07: 800 train questions), the search questions
    are the first N of the group's questions that are not dev questions, in the clusters file's
    order, and the rest of the group is not used; N must fall inside the ordered subset."""
    d = json.loads(Path(path).read_text())
    dev = None
    if dev_split is not None:
        split = json.loads(Path(dev_split).read_text())
        dev, known = set(split["dev"]), set(split["train"]) | set(split["dev"])
        if set(split["train"]) & dev:
            raise SystemExit(f"{dev_split}: the train and dev lists share questions")
    groups = []
    for c in d["clusters"]:
        pool = c["subset"] + [q for q in c["held_out"] if q not in c["subset"]]
        if dev is not None:
            if (out := [q for q in pool if q not in known]):
                raise SystemExit(f"{dev_split}: {len(out)} questions of group {c['cluster']} are in neither list")
            search, held = [q for q in pool if q not in dev], [q for q in pool if q in dev]
            if per_group and len(search) > per_group:      # a cap: the first N non-dev questions
                if sum(q not in dev for q in c["subset"]) < per_group:
                    raise SystemExit(f"group {c['cluster']} has only {sum(q not in dev for q in c['subset'])} "
                                     f"ordered subset questions outside the dev split for {per_group} search "
                                     f"questions; re-run cluster_questions.py with a larger --per-cluster")
                search = search[:per_group]
            if not search or not held:
                raise SystemExit(f"{dev_split}: group {c['cluster']} has {len(search)} search and {len(held)} "
                                 f"held-out questions; both must be non-empty")
            groups.append({"group": c["cluster"], "search": search, "held_out": held,
                           "size": c["size"], "medoid": c.get("medoid"),
                           "medoid_template": c.get("medoid_template", ""),
                           "steps": c.get("steps", {}), "risks": c.get("risks") or c.get("challenges", {}),
                           "knowledge": c.get("knowledge", {}),
                           "ordered": len(c["subset"]) < c["size"]})   # else listed in the dataset's order
            continue
        n = len(pool) if per_group == 0 else min(per_group, len(pool))
        if n < len(pool) and len(c["subset"]) < n:      # a cap must fall inside the ordered subset
            raise SystemExit(f"group {c['cluster']} has only {len(c['subset'])} ordered subset questions for "
                             f"{n} search questions; re-run cluster_questions.py with --per-cluster >= {n}")
        search, held = pool[:n], pool[n:]
        groups.append({"group": c["cluster"], "search": search, "held_out": held,
                       "size": c["size"], "medoid": c.get("medoid"),
                       "medoid_template": c.get("medoid_template", ""),
                       "steps": c.get("steps", {}), "risks": c.get("risks") or c.get("challenges", {}),
                       "knowledge": c.get("knowledge", {})})
    out = {"k": d["k"], "per_group": per_group, "source": str(Path(path).resolve()), "groups": groups}
    if dev_split is not None:
        out["dev_split"] = str(Path(dev_split).resolve())
    return out


def group_profile_text(g: dict, difficulty: dict | None = None) -> str:
    """A group in a few lines, for the model-written seeds and the guided edits."""
    steps = ", ".join(f"{k} ({v:.0%})" for k, v in sorted(g["steps"].items(), key=lambda kv: -kv[1])[:4])
    risks = ", ".join(f"{k} ({v:.0%})" for k, v in sorted(g["risks"].items(), key=lambda kv: -kv[1])[:2])
    know = ", ".join(f"{k} ({v:.0%})" for k, v in sorted(g["knowledge"].items(), key=lambda kv: -kv[1]))
    lines = [f"group {g['group']} ({g['size']} questions): {g['medoid_template']}",
             f"  typical steps: {steps}", f"  main failure risk: {risks}",
             f"  knowledge: {know}"]
    if difficulty:
        lines.append("  dataset difficulty: " + ", ".join(f"{k} {v:.0%}" for k, v in difficulty.items()))
    return "\n".join(lines)


def group_samples_text(g: dict, rows: dict, n: int = 10, max_chars: int = 2500) -> str:
    """`n` of the group's search questions as the debaters see them (question and options),
    without their answers, for the model that writes seeds: the first `n` in the clusters
    file's representative order (the medoid, then alternating far and random picks). A group
    no larger than the file's subset size is listed whole in the dataset's order, which is
    not representative (it runs by discipline), so there the medoid comes first and then a
    fixed random draw of the rest. A question over `max_chars` characters keeps its start and
    its end (where the options are)."""
    qids = list(g["search"])
    med = g.get("medoid")
    # a group listed in the dataset's order gets a fixed random draw, with its medoid first when the
    # medoid is a search question (since 2026-10-07 also when it is not: a held-out medoid left the
    # dataset's order, by discipline: run2's group 2 would have shown 7 Engineering, 2 History, 1 Law)
    if (med in qids and qids[0] != med) or (med not in qids and not g.get("ordered", True)):
        rest = sorted(q for q in qids if q != med)
        random.Random(0).shuffle(rest)
        qids = ([med] if med in qids else []) + rest
    qids = qids[:n]
    return questions_text(qids, rows, f"  example questions ({len(qids)} of the group's {g['size']}; "
                                      f"answers not shown):", max_chars)


def questions_text(qids: list[str], rows: dict, header: str, max_chars: int = 2500) -> str:
    """`qids` as the debaters see them (question and options), without their answers, under
    `header`; '' if there are none. A question over `max_chars` characters keeps its start
    and its end (where the options are)."""
    if not qids:
        return ""
    head = max_chars * 2 // 3
    out = [header]
    for i, q in enumerate(qids, 1):
        text = D.render_question(rows[q]["question"], rows[q].get("options") or []).strip()
        if len(text) > max_chars:
            tail = max_chars - head
            text = f"{text[:head]}\n[... {len(text) - max_chars} characters omitted ...]\n{text[-tail:]}"
        out.append(f"  --- example {i} ---\n" + "\n".join("    " + l for l in text.splitlines()))
    return "\n".join(out)


def difficulty_mix(rows: dict, qids: list[str]) -> dict:
    """Share of each difficulty label among `qids` (empty if the dataset has none, as HLE)."""
    c = Counter(d for q in qids if (d := rows[q].get("difficulty", "?")))
    return {k: v / len(qids) for k, v in c.most_common()} if qids else {}


# --- grammar text for the model -------------------------------------------------------

def grammar_text_v3() -> str:
    """The v3 grammar for the model that writes seeds, built from the live
    vocabulary so it never drifts from the code."""
    names = list(PLAN_ROUNDS)
    kinds = ", ".join(M.ROUND_KINDS)
    if RUN_CAP == MAX_TURNS:                    # one cap (every run before --total-cap)
        cap_text = (f"A question is capped at {MAX_TURNS} turns: an action that would pass the cap is\n"
                    f"replaced by the default stop.")
    else:
        cap_text = (f"The plan (its opening rounds, all of them) may cost at most {MAX_TURNS} turns. Rules\n"
                    f"may run extra rounds after or between them, up to {RUN_CAP} turns for the whole\n"
                    f"question: an action that would pass {RUN_CAP} is replaced by the default stop.")
    judge_line = ("\n    judge                                     chooses one of the answers committed so far"
                  "\n                                              (never a new one); it cannot speak first,"
                  "\n                                              and is never blind"
                  if D.JUDGE_ON else "")
    last_reads = ("  stop:last_commit     stop; answer = the most common letter of the last round that\n"
                  "                       committed one (ties: its earliest speaker)\n"
                  "  stop:last_speaker    the same as stop:last_commit\n"
                  if M.LAST_ROUND_VOTE else
                  "  stop:last_commit     stop; answer = the most recent committed letter\n"
                  "  stop:last_speaker    stop; answer = the last round's letter\n")
    return f"""A program is a JSON object {{"plan": [...], "rules": [...], "default": "stop:<read>"}}.

ROUNDS. Every round, an opening round or an extra round run by a rule, is some speakers at an
effort and a visibility.
  Speakers (the speakers of one round answer independently and never see each other):
    solver, solver_x2, solver_x3, solver_x4   one to four solvers
    expert                                    an expert in the question's field
    expert_solver                             an expert and a solver
    critic                                    looks for a flaw in the answers so far; may correct them
    verifier                                  tests each committed letter against the question's wording
    synthesizer                               weighs the answers so far and commits to one{judge_line}
  Effort: low (the default) or high. A high-effort speaker uses the model's high reasoning
    setting: a long private reasoning, several times slower and dearer. It counts as
    {D.HIGH_COST} turns; a low-effort speaker counts as 1.
  Visibility: by default a speaker sees every earlier answer and its reasoning. A blind
    round (solver and expert rounds only) sees the question and nothing else. The first
    round has nothing to see either way.
  A round is named <speakers>[|high][|blind]. The rounds that exist ({len(names)}; none may cost
  more than {MAX_TURNS} turns):
    {", ".join(names)}

PLAN: 1 to {MAX_PLAN} opening rounds, each written as a JSON spec: {{"personas": [...]}}, plus
"effort": "high" for a high-effort round and "sees": "none" for a blind one. For example
solver_x3 is {{"personas": ["solver", "solver", "solver"]}}, verifier|high is
{{"personas": ["verifier"], "effort": "high"}} and expert_solver|blind is
{{"personas": ["expert", "solver"], "sees": "none"}}. The first plan round never has "sees".
The plan does not start by itself: the action "continue" runs the next plan round, and
when the plan is used up, "continue" behaves like the default stop.

RULES: an ordered list, checked from the top before the first round (at step==0) and again
after every round. The FIRST rule whose conditions all hold decides the next action. If none
holds, the default (always a stop) applies. So some rule must hold at step==0 and start the
debate with "continue" or a round; a program that would stop at step==0 ends before anyone
speaks, with no answer, and is invalid. Most programs begin with
{{"when": ["plan_left"], "do": "continue"}}: as the first rule it runs the plan rounds in order,
and the later rules decide only once the plan is used up. Each rule is
{{"when": [<condition>, ...], "do": <action>}} with 1 or 2 conditions.

CONDITIONS (all about the debate so far, never the answer key):
  step==K / step>=K / step<K / step<=K / step>K   rounds run so far (K in 0..6)
  r1_majority==K / r1_majority>=K / r1_majority<K  size of the largest agreeing block among
                                                   the opening round's letters (1..its width)
  n_distinct==K / n_distinct>=K                    distinct letters committed so far (1..4)
  plan_left           opening rounds remain and no extra action has run yet
  not_extended        no extra (non-plan) action has run yet
  confirmed_switch    the last two speakers agree on a letter that differs from
                      the opening round's majority
  parked              every speaker after the opening round repeated its majority
  last_round_agree    the last round had >= 2 speakers and they all agree
  verifier_backed     the last round's single letter repeats an earlier commit
  kept_answer         the last round's answer is the same as the round before it
                      (each round's answer: its most common letter)
  last:<action>       the previous action was <action> (a round name, or continue)
  ran:<action>        <action> has run at some point
  last_round:<kind>   the last round's speakers were <kind>, at any effort or visibility;
                      kinds: {kinds}
  ran_round:<kind>    a round of <kind> has run at some point
Every condition must be able to hold: last:/ran: only name actions the program's rules
take, last_round:/ran_round: only kinds of round the program has, and r1_majority
cannot exceed the width of the opening round.

ACTIONS:
  continue             the next plan round
  <round name>         one extra round, e.g. critic, verifier|high, solver_x2|blind
{last_reads}  stop:vote            stop; answer = plurality over every letter committed so far

Cost is counted in speaker turns: 1 per low-effort speaker, {D.HIGH_COST} per high-effort
speaker. {cap_text}"""


def _open_wording(text: str) -> str:
    """The grammar text with 'answer' for 'letter' (open answers)."""
    return text.replace("a letter", "an answer").replace("letters", "answers").replace("letter", "answer")


def grammar_text() -> str:
    """The rule and plan grammar, written out for the model that writes seeds
    and guided edits. Built from the live menus so it never drifts from the code."""
    if D.V3:
        return _open_wording(grammar_text_v3()) if D.OPEN else grammar_text_v3()
    acts = ", ".join(sorted(M.ACTIONS))
    kinds = ", ".join(M.ROUND_KINDS)
    plans = "\n".join(f"    {name}: {json.dumps(spec)}" for name, spec in PLAN_ROUNDS.items())
    return f"""A program is a JSON object {{"plan": [...], "rules": [...], "default": "stop:<read>"}}.

PLAN: 1 to {MAX_PLAN} opening rounds. Each must be exactly one of these specs
(a "sees" key controls what the speakers are shown of the transcript):
{plans}
The action "continue" runs the next plan round; when the plan is used up,
"continue" behaves like the default stop.

RULES: an ordered list. After every round the rules are checked from the top
and the FIRST rule whose conditions all hold decides the next action. If none
holds, the default (always a stop) applies. Each rule is
{{"when": [<condition>, ...], "do": <action>}} with 1 or 2 conditions.

CONDITIONS (all about the debate so far, never the answer key):
  step==K / step>=K / step<K / step<=K / step>K   rounds run so far (K in 0..6)
  acts==K / acts>=K                                actions taken so far
  r1_majority==K / r1_majority>=K / r1_majority<K  size of the largest agreeing
                                                   block among the opening round's letters (1..4)
  n_distinct==K / n_distinct>=K                    distinct letters committed so far (1..4)
  plan_left           opening rounds remain and no extra action has run yet
  not_extended        no extra (non-plan) action has run yet
  confirmed_switch    the last two speakers agree on a letter that differs from
                      the opening round's majority
  parked              every speaker after the opening round repeated its majority
  last_round_agree    the last round had >= 2 speakers and they all agree
  verifier_backed     the last round's single letter repeats an earlier commit
  kept_answer         the last round's answer is the same as the round before it
                      (each round's answer: its most common letter)
  last:<action>       the previous action was <action> (actions listed below, or continue)
  ran:<action>        <action> has run at some point
  last_round:<kind>   the last round's speakers were <kind>; kinds: {kinds}
  ran_round:<kind>    a round of <kind> has run at some point

ACTIONS (extra rounds beyond the plan): {acts}
  critic               one critic who sees everything and may correct the answer
  verifier             one verifier who tests each committed letter against the question
  fresh                two independent solvers who see nothing of the debate
  expert_blind         one field expert who sees nothing of the debate
  pair_expert_solver   an expert and a solver who see only the last round
  synthesizer          one synthesizer who weighs everything and commits
  continue             the next plan round
  stop:last_commit     stop; answer = the most recent committed letter
  stop:last_speaker    stop; answer = the last round's letter
  stop:vote            stop; answer = plurality over every letter committed so far

Cost is counted in speaker turns; each speaker is one turn. A question is
capped at 16 turns, after which the program is stopped by its default."""

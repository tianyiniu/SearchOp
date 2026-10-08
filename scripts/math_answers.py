"""Answers to MATH questions under --answers math (2026-10-07): when two answers are the same answer,
and grading. debate_mcq reads a speaker's answer (the last \\boxed{...} or ANSWER line of its text) and
gives it the key below; every reader of an answer (votes, the rules, the final read, the summary check)
then compares keys.

The key of an answer:
  - an answer with no variables (a number, an interval, a tuple, a set, a matrix of numbers): the value
    worked out exactly, as LaTeX. Decimals the speaker typed count as the exact fractions they write
    (0.75 is 3/4), then the value is evaluated and multiplied out: '\\sqrt{8}', '2\\sqrt{2}' and
    '\\sqrt{8.0}' all get '2 \\sqrt{2}'.
  - any other answer: its own spelling with the formatting removed (\\left / \\right, \\dfrac and
    \\tfrac, \\leq / \\geq, spaces next to a non-letter). It is never rewritten as other algebra: later
    speakers are shown the key, and a factored answer must stay factored.
A worked-out key is used only if math-verify says it equals the answer and the key of the key is itself;
the formatted spelling only if math-verify reads it as it reads the answer. Otherwise the answer keeps
its spelling. So two answers with one key are always equal by math-verify; the converse fails only for
answers that differ as algebra (-n(2n+1) and -2n^2-n), which then count apart in a vote. Measured on
929 distinct answers of Qwen3.5-4B replies to MATH Level 5 test questions: no pair merged that
math-verify calls different, 21 of 364 pairs it calls equal left apart, every key stable, and grading
the key gave math-verify's verdict on the answer itself for all 929.

Grading: math-verify, exactly as the external baselines grade (baselines/tasks.correct): the key
'\\boxed{<answer>}' parsed and verified against the key of the question, with math-verify's defaults.

math-verify limits its time with signal alarms, which work only in a process's main thread, and the
executor runs debates in threads (in a thread, math-verify raises, and with raise_on_error off it
would quietly read no answer at all). So every math-verify call runs in a small pool of worker
processes, in their main thread, with math-verify's own time limits; the results are cached here, so
each distinct answer is worked out once per process.
"""

from __future__ import annotations

import multiprocessing
import os
import re
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

WORKERS = 4                  # worker processes
CALL_TIMEOUT = 120.0         # seconds to wait for one worker call (math-verify's own limit is 5 s per step)

# --- in the worker processes ------------------------------------------------------------------------


def _parse(a: str):
    from math_verify import parse
    return parse("\\boxed{" + a + "}")               # as baselines/tasks._math


def _fmt(a: str) -> str:
    """Formatting only (see the module docstring)."""
    b = re.sub(r"\\(?:left|right)(?![a-zA-Z])", "", a)
    b = re.sub(r"\\[dt]frac(?![a-zA-Z])", r"\\frac", b)
    b = re.sub(r"\\(le|ge)q(?![a-zA-Z])", r"\\\1", b)
    b = re.sub(r"\s*([^A-Za-z\s])\s*", r"\1", b)     # spaces between letters can end a command: kept
    return re.sub(r"\s+", " ", b).strip()


def _worked_out(parsed):
    """The exact value of a parsed answer with no variables, as LaTeX; None for any other answer."""
    import sympy
    if not parsed or not isinstance(parsed[0], sympy.Basic) or parsed[0].free_symbols:
        return None
    obj = parsed[0]
    obj = obj.xreplace({f: sympy.Rational(str(f)) for f in obj.atoms(sympy.Float)})
    obj = obj.doit()
    if isinstance(obj, sympy.Expr):
        obj = sympy.expand(obj)
    return sympy.latex(obj)


_TEXT_WRAP = re.compile(r"\\(?:text|textbf|mathrm)\{(.*)\}", re.S)


def _same(parsed, other: str) -> bool:
    """Whether `other` reads as the answer `parsed` was parsed from (math-verify; an answer it cannot
    parse is never replaced)."""
    from math_verify import verify
    po = _parse(other)
    return bool(parsed) and bool(po) and verify(parsed, po)


def _key_core(a: str) -> str:
    """The key candidate of `a`: its worked-out value, else its spelling without a \\text{} wrapper and
    formatting, else its formatted spelling, else itself; each only if math-verify reads it as `a`."""
    parsed = _parse(a)
    try:
        k = _worked_out(parsed)
    except Exception:
        k = None
    if k is not None and _same(parsed, k):
        return k
    if (m := _TEXT_WRAP.fullmatch(a)) and m.group(1).strip():
        for u in (_fmt(m.group(1).strip()), m.group(1).strip()):
            if _same(parsed, u):
                return u
    f = _fmt(a)
    if f != a and _same(parsed, f):
        return f
    return a


def _key_in_worker(a: str) -> str:
    """_key_core, kept only if it is stable (the key of the key is the key); else the answer itself."""
    try:
        k = _key_core(a)
        if k == a or _key_core(k) == k:
            return k
    except Exception:
        pass
    return a


def _grade_in_worker(gold: str, answer: str) -> bool:
    from math_verify import verify
    try:
        return bool(verify(_parse(gold), _parse(answer)))
    except Exception:
        return False


# --- in the executor's process ----------------------------------------------------------------------

_pool: ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()
_keys: dict[str, str] = {}
_grades: dict[tuple[str, str], bool] = {}
_cache_lock = threading.Lock()
STATS = {"keys": 0, "grades": 0, "worker_failures": 0}


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            # spawn, not fork: the executor's process runs many threads, and a forked child of a
            # threaded process can deadlock. A spawned worker first runs the parent's main script again,
            # by its file name. A script read from stdin ('python -', as the pipeline's shell steps
            # run their checks) has the file name '<stdin>', which names no file, so every worker
            # would fail and every key and grade fall back (seen in a test run, 2026-10-07). Such a
            # name is dropped: the workers then run no main script, which they do not need.
            main = sys.modules.get("__main__")
            path = getattr(main, "__file__", None)
            if path is not None and not os.path.isfile(os.path.join(os.getcwd(), path)):
                del main.__file__
            _pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn"))
        return _pool


def _reset_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def _call(fn, *args) -> tuple[bool, object]:
    """(True, fn(*args)) run in a worker process; (False, None) if it fails twice or takes too long.
    A failure is counted and printed, never cached: the next call tries again."""
    for _ in range(2):
        try:
            return True, _get_pool().submit(fn, *args).result(timeout=CALL_TIMEOUT)
        except FutureTimeout:
            break                                    # a worker stuck past math-verify's own limits
        except Exception:                            # a broken pool (a worker died): start a new one
            _reset_pool()
    with _cache_lock:
        STATS["worker_failures"] += 1
    print(f"math_answers: {fn.__name__}{args!r} failed in the worker processes "
          f"({STATS['worker_failures']} failures so far)", file=sys.stderr, flush=True)
    return False, None


def key(answer: str) -> str:
    """The key of an answer (an answer as debate_mcq.normalize_answer leaves it)."""
    with _cache_lock:
        if answer in _keys:
            return _keys[answer]
    ok, k = _call(_key_in_worker, answer)
    if not ok:
        return answer                                # its own spelling, this time only
    with _cache_lock:
        STATS["keys"] += 1
        return _keys.setdefault(answer, k)


def grade(gold: str, answer: str | None) -> bool:
    """Whether `answer` is right for a question whose key is `gold` (math-verify)."""
    if answer is None:
        return False
    with _cache_lock:
        if (gold, answer) in _grades:
            return _grades[(gold, answer)]
    ok, right = _call(_grade_in_worker, gold, answer)
    if not ok:
        return False                                 # counted and printed by _call; not cached
    with _cache_lock:
        STATS["grades"] += 1
        return _grades.setdefault((gold, answer), right)


def grade_row(row: dict, answer: str | None) -> bool:
    """grade() for a dataset row (its key in 'answer'): evolve_program_mcq's grader under --answers math."""
    return grade(row["answer"], answer)

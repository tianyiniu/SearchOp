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


def configure_executor(digest_head: int = DIGEST_HEAD, digest_tail: int = DIGEST_TAIL,
                       visible_reasoning: bool = False, deep_think: bool = False,
                       summary_words: int = 120) -> dict:
    """Switch the shared executor to the new pipeline's settings. Returns the
    settings as a dict, which the archive header records and a resume checks.
    `visible_reasoning` is for models that think in a hidden channel (gpt-oss);
    the settings dict names it only when on, so archives made without it still
    match on resume."""
    if digest_tail <= 0:
        raise SystemExit("the v2 executor needs a digest tail > 0")
    D.set_digest(digest_head, digest_tail)
    SF.set_v2(True)
    M.drop_eliminator()
    D.set_round_parallel(True)        # speakers of one round in flight together
    settings = {"v2": True, "digest": [digest_head, digest_tail], "answer_tokens": ANSWER_TOKENS,
                "eliminator": False}
    if visible_reasoning:
        SF.set_visible_reasoning(True)
        settings["visible_reasoning"] = True
    # Both are named in the settings only when on, like visible_reasoning, so
    # archives made without them still match on resume.
    if summary_words != 120:
        D.set_summary_words(summary_words)
        settings["summary_words"] = summary_words
    if deep_think:
        D.set_deep_think(True)
        M.add_deep_think()
        PLAN_ROUNDS[D.DEEP_PERSONA] = {"personas": [D.DEEP_PERSONA]}
        # one deep-think turn and nothing else: the "just think longer" program. It
        # is a baseline beside direct, and among the seeds it takes direct's place
        # (see seed_protocols), so the number of seeds does not change.
        PROTOCOLS["deep_direct"] = {"plan": [{"personas": [D.DEEP_PERSONA]}], "rules": [CONT()],
                                    "default": "stop:last_commit"}
        settings["deep_think"] = {"cost": D.DEEP_COST}
    return settings


def seed_protocols() -> dict[str, dict]:
    """The literature programs used as seeds, in their fixed order. With
    deep-think on, the one-solver program (direct) is replaced, in place, by the
    one-deep-thinker program (deep_direct); both stay in PROTOCOLS as baselines."""
    if "deep_direct" not in PROTOCOLS:
        return dict(PROTOCOLS)
    return {("deep_direct" if name == "direct" else name): (PROTOCOLS["deep_direct"] if name == "direct" else prog)
            for name, prog in PROTOCOLS.items() if name != "deep_direct"}


def add_executor_args(ap) -> None:
    """--deep-think and --summary-words, the same on every entry script."""
    ap.add_argument("--deep-think", action="store_true",
                    help="add the deep-think speaker (highest reasoning setting, no reply cap) as a "
                         "plan round and a move, and deep_direct as a literature program")
    ap.add_argument("--summary-words", type=int, default=120,
                    help="word limit of the summary later speakers read (120 = the original). Above 120, a "
                         "reply within the limit is shown as it is, through the digest window, so widen "
                         "--digest-head/--digest-tail to hold a reply of that length (~7 characters a word)")


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
    return out


def normalize_program(prog: dict) -> dict:
    """The same program with every plan spec in normalised form and the rule
    list deep-copied, so the text is canonical before it is hashed."""
    return {"plan": [norm_spec(s) for s in prog["plan"]],
            "rules": [{"when": list(r["when"]), "do": r["do"]} for r in prog["rules"]],
            "default": prog["default"]}


def plan_round_name(spec: dict) -> str:
    spec = norm_spec(spec)
    for name, s in PLAN_ROUNDS.items():
        if s == spec:
            return name
    return "+".join(spec["personas"]) + (f"|{spec['sees']}" if spec.get("sees") else "")


def validate_program(prog: dict) -> None:
    """The rule grammar's validation plus the plan grammar's."""
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


# --- random programs and mutation ----------------------------------------------

def random_plan(rng: random.Random) -> list[dict]:
    plan = [solvers(rng.choice(OPENING_WIDTHS))]
    while len(plan) < MAX_PLAN and rng.random() < 0.35:
        plan.append(copy_program(rng.choice(list(PLAN_ROUNDS.values()))))
    return plan


def random_program(rng: random.Random, min_rules: int = 3, max_rules: int = 8) -> dict:
    """A program drawn from the grammar. Most such programs are useless in a
    predictable way (stop at once, or run to the cap); the seed stage's sanity
    run filters those. A plan_left->continue rule is included most of the time,
    otherwise the opening rounds would rarely run at all."""
    rules = [{"when": [M.random_condition(rng) for _ in range(rng.choice([1, 1, 2]))],
              "do": M.random_action(rng)}
             for _ in range(rng.randint(min_rules, max_rules))]
    if rng.random() < 0.7:
        rules.insert(rng.randrange(len(rules) + 1), CONT())
    prog = {"plan": random_plan(rng), "rules": rules,
            "default": "stop:" + rng.choice(M.STOP_READS)}
    validate_program(prog)
    return prog


PLAN_OPS = ("plan_width", "plan_replace", "plan_add", "plan_drop")


def mutate_plan(prog: dict, rng: random.Random, op: str | None = None) -> tuple[dict, str]:
    """One edit to the opening rounds. Returns (child, op name). With `op`
    None the kind is drawn here (the v2 search); the v3 search names it."""
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
    elif op == "plan_replace":
        i = rng.randrange(len(plan))
        choices = [s for s in PLAN_ROUNDS.values() if s != plan[i]]
        plan[i] = copy_program(rng.choice(choices))
    elif op == "plan_add":
        plan.insert(rng.randrange(len(plan) + 1), copy_program(rng.choice(list(PLAN_ROUNDS.values()))))
    else:
        plan.pop(rng.randrange(len(plan)))
    return p, op


RULE_OPS = ("change_action", "add_rule", "change_cond", "change_default", "drop_rule", "swap_rules")
ALL_OPS = RULE_OPS + ("plan",)


def mutate_rules(prog: dict, rng: random.Random, op: str) -> tuple[dict, str]:
    """One rule edit of the named kind (evolve_program_mcq's operators, but with
    the kind chosen by the caller so the guided mutator and the logs can name
    it). Falls back to the closest legal kind when the named one cannot apply;
    returns (child, the kind actually applied)."""
    p = copy_program(prog)
    rules = p["rules"]
    if op in ("drop_rule", "swap_rules") and len(rules) < 2:
        op = "change_action"
    if op == "add_rule" and len(rules) >= M.MAX_RULES:
        op = "change_cond"
    if op == "change_action":
        rng.choice(rules)["do"] = M.random_action(rng)
    elif op == "add_rule":
        rule = {"when": [M.random_condition(rng) for _ in range(rng.choice([1, 1, 2]))],
                "do": M.random_action(rng)}
        rules.insert(rng.randrange(len(rules) + 1), rule)
    elif op == "change_cond":
        rule = rng.choice(rules)
        if rule["when"] and rng.random() < 0.7:
            rule["when"][rng.randrange(len(rule["when"]))] = M.random_condition(rng)
        else:
            rule["when"] = rule["when"] + [M.random_condition(rng)]
    elif op == "change_default":
        p["default"] = "stop:" + rng.choice([r for r in M.STOP_READS
                                              if "stop:" + r != p["default"]])
    elif op == "drop_rule":
        rules.pop(rng.randrange(len(rules)))
    elif op == "swap_rules":
        i, j = rng.sample(range(len(rules)), 2)
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


# The v3 search has no edit-type probabilities: one kind is drawn uniformly
# from the kinds that can apply to the program at hand.
EDIT_KINDS = RULE_OPS + PLAN_OPS


def applicable_edits(prog: dict) -> list[str]:
    """The edit kinds that can apply to `prog`, in EDIT_KINDS order."""
    n_rules, n_plan = len(prog["rules"]), len(prog["plan"])
    out = []
    for op in EDIT_KINDS:
        if op in ("drop_rule", "swap_rules") and n_rules < 2:
            continue
        if op == "add_rule" and n_rules >= M.MAX_RULES:
            continue
        if op == "plan_add" and n_plan >= MAX_PLAN:
            continue
        if op == "plan_drop" and n_plan <= 1:
            continue
        out.append(op)
    return out


def mutate_uniform(prog: dict, rng: random.Random) -> tuple[dict, str]:
    """One random edit whose kind is drawn uniformly from the applicable kinds.
    Returns (child, op name); an invalid or unchanged child is redrawn."""
    kinds = applicable_edits(prog)
    for _ in range(50):
        op = rng.choice(kinds)
        child, name = mutate_plan(prog, rng, op) if op in PLAN_OPS else mutate_rules(prog, rng, op)
        try:
            validate_program(child)
        except (AssertionError, ValueError):
            continue
        if canon(child) != canon(prog):
            return child, name
    return copy_program(prog), "none"


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


def behaviour_hash(rec: dict[str, list]) -> str:
    """Identity of a fully covered record: what it did and what it answered on
    every question. Two programs with the same hash are the same individual."""
    blob = json.dumps(sorted((q, v[2], v[3]) for q, v in rec.items()), separators=(",", ":"))
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
        # free-form provenance a search may attach (the v3 search records the
        # slot and target group a child was bred for, and its screen verdict)
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
        self.reps.setdefault(rep, {})[qid] = [int(bool(out["correct"])), out["n_calls"],
                                              out["letter"] or "?", path_of(out)]

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

def load_groups(path: Path, per_group: int) -> dict:
    """The search questions and held-out questions per group. `subset` lists
    in the clusters file are ordered (centre, then alternating farthest and
    random picks) and the order does not depend on how many were requested,
    so the first `per_group` of each is the search set at that size and any
    later question of the group is held out."""
    d = json.loads(Path(path).read_text())
    groups = []
    for c in d["clusters"]:
        if len(c["subset"]) < per_group:
            raise SystemExit(f"group {c['cluster']} has only {len(c['subset'])} subset questions; "
                             f"re-run cluster_questions.py with --per-cluster >= {per_group}")
        search = c["subset"][:per_group]
        held = c["subset"][per_group:] + [q for q in c["held_out"] if q not in search]
        groups.append({"group": c["cluster"], "search": search, "held_out": held,
                       "size": c["size"], "medoid_template": c.get("medoid_template", ""),
                       "steps": c.get("steps", {}), "risks": c.get("risks", {}),
                       "knowledge": c.get("knowledge", {})})
    return {"k": d["k"], "per_group": per_group, "source": str(Path(path).resolve()),
            "groups": groups}


def group_profile_text(g: dict, difficulty: dict | None = None) -> str:
    """A group in a few lines, for the model-written seeds and the guided edits."""
    steps = ", ".join(f"{k} ({v:.0%})" for k, v in sorted(g["steps"].items(), key=lambda kv: -kv[1])[:4])
    risks = ", ".join(f"{k} ({v:.0%})" for k, v in sorted(g["risks"].items(), key=lambda kv: -kv[1])[:2])
    know = ", ".join(f"{k} ({v:.0%})" for k, v in sorted(g["knowledge"].items(), key=lambda kv: -kv[1]))
    lines = [f"group {g['group']} ({g['size']} questions): {g['medoid_template']}",
             f"  typical moves: {steps}", f"  main failure risk: {risks}",
             f"  knowledge: {know}"]
    if difficulty:
        lines.append("  dataset difficulty: " + ", ".join(f"{k} {v:.0%}" for k, v in difficulty.items()))
    return "\n".join(lines)


def difficulty_mix(rows: dict, qids: list[str]) -> dict:
    c = Counter(rows[q].get("difficulty", "?") for q in qids)
    return {k: v / len(qids) for k, v in c.most_common()} if qids else {}


# --- grammar text for the model -------------------------------------------------------

def grammar_text() -> str:
    """The rule and plan grammar, written out for the model that writes seeds
    and guided edits. Built from the live menus so it never drifts from the code."""
    acts = ", ".join(sorted(M.ACTIONS))
    kinds = ", ".join(M.ROUND_KINDS)
    plans = "\n".join(f"    {name}: {json.dumps(spec)}" for name, spec in PLAN_ROUNDS.items())
    # the deep-think lines appear only when the speaker is switched on, so the
    # text every earlier run showed the model is unchanged
    deep_act = (f"\n  {D.DEEP_PERSONA:<20} one deep thinker who sees everything, is given the model's highest"
                f"\n  {'':<20} reasoning setting and no reply limit; costs {D.DEEP_COST} turns"
                if D.DEEP_THINK else "")
    deep_cost = (f" The one exception is a {D.DEEP_PERSONA} speaker, which counts as {D.DEEP_COST} turns: it is"
                 f" several times slower and dearer than any other speaker, so use it where it matters."
                 if D.DEEP_THINK else "")
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
  synthesizer          one synthesizer who weighs everything and commits{deep_act}
  continue             the next plan round
  stop:last_commit     stop; answer = the most recent committed letter
  stop:last_speaker    stop; answer = the last round's letter
  stop:vote            stop; answer = plurality over every letter committed so far

Cost is counted in speaker turns; each speaker is one turn.{deep_cost} A question is
capped at 16 turns, after which the program is stopped by its default."""

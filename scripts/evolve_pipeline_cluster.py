"""Per-group evolutionary search for debate-control programs: the search of the cluster pipeline
(run_pipeline_cluster.sh; named evolve_program_clusters_v3.py until 2026-10-06).

One search over the k question groups of a clusters file (outputs/describe_v3/clusters_600_train.json
in the cluster pipeline). Every count is tied to k; nothing is split
by hand-set shares.

Slots. Each group keeps two programs, so there are 2k parents:
    A  strongest    the highest score on the group; ties go to fewer turns.
                    With --tie-questions t (0 by default), every program within
                    t questions of the best score ties with it, so the
                    cheapest of them (fewest turns) holds the slot.
    B  specialist   the largest rank gap. Scores are turned into percentile
                    ranks inside each group; a program's gap on a group is its
                    rank there minus its highest rank on any other group.
                    Largest gap wins, ties go to fewer turns, and the group's
                    slot-A holder is skipped.
Only programs scored on every search question can hold a slot. The archive
only grows: a child never replaces its parent, it competes with it.

Confirmation. Every program is first scored once (replicate 0) on every
search question. A program that would take a slot is scored a second time
(replicate 1) on that slot's questions (its group's; all of them for the
global slot); its score there becomes the two-replicate mean and the slots
are worked out again, until every holder has both replicates. A program's
score on a group is always the mean over the replicates it has.

Global slot. One more slot, G, holds the strongest program over ALL search
questions (ties to fewer turns, by the same rule as slot A); it breeds like any other slot, with all the
search questions as its group. So there are 2k + 1 parents.

Children. Each slot's holder has two children per generation (4k + 2 in all).
Before it breeds, a holder is pruned: rules that never decided a step on any
recorded question are dropped (that changes nothing on those questions). A
child is one random edit of the pruned holder. A family of edits is drawn evenly
from the families that apply, then a kind evenly within it (program_space.mutate_by_family,
since 2026-10-07; every kind evenly before): the rule edits (the rule an edit touches is
picked in proportion to how many steps it decided, plus one), the plan edits (add, replace or
drop a round), the width edit (with --any-round-width, set_width in place of plan_width: the
number of solvers of any solver round), switching one round's effort, switching one round's
visibility, and crossover (this holder's opening rounds with another holder's rules, or the
other way round). A child whose first round has a critic, verifier or synthesizer is drawn
again (2026-10-07). It is first replayed against the recordings, which is free,
and drawn again if its plan rounds cost more than the turn cap (its last plan
round could never run), if its text is already known, if it behaves exactly like its
parent or its sibling on the slot's group, or exactly like an archived program
overall. Then it is scored once (replicate 0) on every search question, so
every child can take any slot, not only the one it was bred for.

Stop. A fixed number of generations (10): 10 x (4k + 2) new programs.

    python scripts/evolve_pipeline_cluster.py \\
        --seeds outputs/pipeline_cluster/seeds.json --out outputs/pipeline_cluster/run1
    python scripts/evolve_pipeline_cluster.py --out outputs/pipeline_cluster/run1 --resume

    # champions on held-out questions (two replicates), with the literature
    # baselines scored on the same questions
    python scripts/evolve_pipeline_cluster.py --out outputs/pipeline_cluster/run1 --pick-champions
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_space as P  # noqa: E402
import schema_fitness as SF  # noqa: E402
from program_space import ProgRecord  # noqa: E402

ROOT = P.ROOT
SLOT_KINDS = ("A", "B")            # strongest, specialist (per group)
GLOBAL = "all"                     # the pseudo-group of the global slot: every search question
GLOBAL_SLOT = "G"
CHILDREN_PER_SLOT = 2
QUICK_ATTEMPTS = 3                 # tries of a failed debate before the search asks whether the server is up
SLOW_WAITS = (30, 120, 300, 600)   # seconds before each further try while the server answers
OUTAGE_WAIT = 7200                 # seconds a search waits for a server that does not answer
MAX_ATTEMPTS = 12                  # tries of a failed debate in all, waits included
MAX_REDRAWS = 200                  # a guard against a parent with no non-neutral edit; never binds in practice
EPS = 1e-9


def log(msg: str = "") -> None:
    SF.log(msg)


# --- small statistics ------------------------------------------------------------------

def paired_se(a: list[float], b: list[float]) -> tuple[float, float]:
    """(mean, standard error) of the paired differences a[i] - b[i]."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return (d[0] if d else 0.0), 0.0
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    return mean, math.sqrt(var / n)


def percentile_ranks(scores: dict[str, float]) -> dict[str, float]:
    """Percentile rank in (0, 1) of each key, higher score = higher rank; tied
    scores share the average of their ranks."""
    n = len(scores)
    order = sorted(scores, key=lambda k: scores[k])
    out: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(scores[order[j + 1]] - scores[order[i]]) < EPS:
            j += 1
        avg_rank = (i + j) / 2 + 1                      # ranks are 1..n
        for key in order[i: j + 1]:
            out[key] = (avg_rank - 0.5) / n
        i = j + 1
    return out


def server_up(args) -> bool:
    """Whether the model server answers (its /models lists the model). A function of the run's
    arguments, not a method: eval_routed_dev calls Search.run_jobs on a stand-in object."""
    return P.server_window(args.base_urls, args.model, getattr(args, "api_key", "EMPTY")) is not None


def end_line(path: Path) -> None:
    """End a JSONL file's torn last line (a kill during a write) with a newline, so the next record
    starts on its own line instead of joining the torn one and being lost with it."""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as fh:
            fh.seek(-1, 2)
            last = fh.read(1)
        if last != b"\n":
            with path.open("a") as fh:
                fh.write("\n")


def tie_questions(args) -> int:
    """--tie-questions (0 when the namespace has none: an evaluation script's)."""
    return int(getattr(args, "tie_questions", 0) or 0)


def strongest(pool: list, score, turns, n: int, tie: int):
    """The strongest of `pool`: the best score, ties to fewer turns, then the older program (then
    the name). With `tie` > 0 a tie is wider: every program within `tie` questions of the best score
    (of `n` questions) ties with it, so the fewest turns among them wins (then the higher score, the
    older program, the name). The comparison is always with the best score itself, so a chain of
    small losses cannot add up. With `tie` 0 this is exactly the earlier rule."""
    if tie <= 0:
        return min(pool, key=lambda r: (-round(score(r), 9), turns(r), r.gen, r.name))
    floor = round(max(score(r) for r in pool) - tie / n, 9)
    near = [r for r in pool if round(score(r), 9) >= floor]
    return min(near, key=lambda r: (turns(r), -round(score(r), 9), r.gen, r.name))


class Search:
    # Debates still failing after the quick tries: wait for the server or stop the search nonzero
    # (run_jobs), and on resume complete any program archived with debates missing (load_archive).
    # The global search sets it False: it completes or stops on its own (complete_short, finish_or_stop).
    WAIT_FOR_SERVER = True
    # On resume, the children of a generation stopped part way are dropped and it is drawn again
    # (load_archive). The global search sets it False: it takes those children as they are (its redo).
    DROP_STOPPED_GENERATION = True

    def __init__(self, args, rows: dict, groups: dict, runner, settings: dict,
                 seed_source: dict[str, str] | None = None):
        self.args, self.rows, self.groups, self.runner = args, rows, groups, runner
        self.settings = settings
        self.gq: dict = {g["group"]: list(g["search"]) for g in groups["groups"]}
        self.group_ids = [g["group"] for g in groups["groups"]]
        self.qids: list[str] = [q for g in self.group_ids for q in self.gq[g]]
        self.gq[GLOBAL] = self.qids
        self.seed_source: dict[str, str] = dict(seed_source or {})
        self.archive: dict[str, ProgRecord] = {}
        self.by_behaviour: dict[str, str] = {}
        self.slots: dict[tuple[int, str], str] = {}
        self.gaps_b: dict[tuple[int, str], float] = {}      # (group, key) -> rank gap, last computed
        self.neutral: set[tuple[str, int]] = set()          # (program text, group) known to be neutral
        self.gen = 0
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        self.archive_path = out / "archive.jsonl"
        self.log_path = out / "generations.jsonl"
        self._archive_fh = None

    # ------------------------------------------------------------------ archive
    def header(self) -> dict:
        h = {"header": True, "version": 3, "settings": self.settings,
             "clusters": self.groups["source"], "per_group": self.groups["per_group"],
             "k": self.groups["k"], "qids": self.qids,
             "groups": {str(g): self.gq[g] for g in self.group_ids},
             "seed": self.args.seed, "seed_source": self.seed_source}
        if tie_questions(self.args):                    # named only when used, so older archives match
            h["tie_questions"] = tie_questions(self.args)
        return h

    def open_archive(self, resume: bool) -> None:
        if resume:
            self.load_archive()
            return
        if self.archive_path.exists():
            raise SystemExit(f"{self.archive_path} exists; pass --resume to continue it "
                             f"or choose another --out")
        self._archive_fh = self.archive_path.open("a")
        self._archive_fh.write(json.dumps(self.header()) + "\n")
        self._archive_fh.flush()

    def load_archive(self) -> None:
        if not self.archive_path.exists():
            raise SystemExit(f"--resume: {self.archive_path} does not exist")
        header = None
        recs: dict[str, dict] = {}
        for line in self.archive_path.open():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue                                    # a torn last line
            if d.get("header"):
                header = d
                continue
            recs[d["key"]] = d                              # the last record of a key wins
        if header is None:
            raise SystemExit(f"{self.archive_path} has no header line")
        mine = self.header()
        for field in ("version", "settings", "clusters", "k", "per_group", "qids", "tie_questions"):
            if header.get(field) != mine.get(field):
                raise SystemExit(f"--resume: the archive's {field} differs from this run's "
                                 f"(a cluster search has fixed settings and a fixed question set)")
        if not self.seed_source:
            self.seed_source = dict(header.get("seed_source") or {})
        # the last COMPLETE generation is the last one logged
        if self.log_path.exists():
            for line in self.log_path.open():
                try:
                    self.gen = max(self.gen, int(json.loads(line).get("gen", 0)))
                except (ValueError, TypeError):
                    pass
        # A generation stopped part way (kill, or the server down past OUTAGE_WAIT) is drawn again from
        # its start: its children are dropped from the archive (the file as it was is kept beside it),
        # and the redone generation draws and runs new ones (any debate already recorded replays free).
        stale = {k for k, d in recs.items() if d.get("gen", 0) > self.gen} if self.DROP_STOPPED_GENERATION else set()
        if stale:
            self.drop_from_archive(stale)
            for k in stale:
                del recs[k]
        for d in recs.values():
            rec = ProgRecord.from_json(d)
            rec.dup_of = None
            self.archive[rec.key] = rec
        for rec in self.archive.values():                   # file order = generation order
            self.mark_duplicate(rec)
        end_line(self.archive_path)                         # a torn last line must not swallow the next record
        self._archive_fh = self.archive_path.open("a")
        log(f"resumed {len(self.archive)} programs from {self.archive_path} "
            f"(last complete generation {self.gen})")
        # a program archived with debates missing (server errors, before 2026-10-06 nothing filled them
        # and such a program could never hold a slot) is completed now
        incomplete = [r for r in self.archive.values() if r.gaps(self.qids, 0)] if self.WAIT_FOR_SERVER else []
        if incomplete:
            log(f"  {len(incomplete)} archived programs miss debates; running them now")
            self.run_jobs([(r, q, 0) for r in incomplete for q in r.gaps(self.qids, 0)], "filling missing debates")
            for r in incomplete:
                self.save(r)
                self.mark_duplicate(r)

    def drop_from_archive(self, keys: set[str]) -> None:
        """Rewrite the archive without the records of `keys` (the children of a generation stopped
        part way), in file order; the file as it was is kept as archive.jsonl.stopped_gen<n>."""
        lines = self.archive_path.read_text().splitlines()
        kept = []
        for line in lines:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue                                    # a torn line
            if d.get("header") or d.get("key") not in keys:
                kept.append(line)
        backup = self.archive_path.with_name(f"{self.archive_path.name}.stopped_gen{self.gen + 1}")
        shutil.copy2(self.archive_path, backup)
        tmp = self.archive_path.with_name(self.archive_path.name + ".tmp")
        tmp.write_text("\n".join(kept) + "\n")
        os.replace(tmp, self.archive_path)
        log(f"generation {self.gen + 1} was stopped part way: its {len(keys)} children are dropped and it is "
            f"drawn again (the archive as it was: {backup.name})")

    def save(self, rec: ProgRecord) -> None:
        self._archive_fh.write(json.dumps(rec.to_json()) + "\n")
        self._archive_fh.flush()

    def add(self, rec: ProgRecord) -> None:
        self.archive[rec.key] = rec
        self.mark_duplicate(rec)
        self.save(rec)

    def mark_duplicate(self, rec: ProgRecord) -> None:
        """A fully scored program whose behaviour matches an earlier program's
        is a duplicate individual: kept on file, never selected."""
        h = rec.behaviour(self.qids)
        if h is None:
            return
        owner = self.by_behaviour.setdefault(h, rec.key)
        rec.dup_of = None if owner == rec.key else owner

    def eligible(self) -> list[ProgRecord]:
        """Programs that may hold a slot: scored on every search question and
        not a behavioural duplicate of an earlier program."""
        return [r for r in self.archive.values() if r.dup_of is None and not r.gaps(self.qids, 0)]

    def source_of(self, rec: ProgRecord) -> str:
        return self.seed_source.get(rec.lineage, "?")

    # ------------------------------------------------------------------ scoring
    def replay(self, rec: ProgRecord, qids: list[str], rep: int = 0) -> None:
        """Cache-only pass: nothing is spent."""
        self.runner.reset_budget(0)
        for q in rec.gaps(qids, rep):
            try:
                rec.record(rep, q, M.run_program(rec.program, self.runner, self.rows[q], rep=rep,
                                                 max_calls=self.args.max_calls_per_question))
            except M.OffCache:
                pass

    def run_jobs(self, jobs: list[tuple[ProgRecord, str, int]], label: str) -> int:
        """Many (program, question, replicate) debates in one thread pool, so
        the server is kept busy whatever the mix of programs. Each result is
        recorded on its program. A debate that fails on a server error is tried
        again: QUICK_ATTEMPTS quick tries, then slow ones (SLOW_WAITS) while the server
        answers, or a wait of up to OUTAGE_WAIT while it is down, MAX_ATTEMPTS tries in
        all; then, with WAIT_FOR_SERVER, the search stops nonzero (without it, the debate is left
        missing). Returns the speaker turns spent."""
        todo, seen = [], set()
        for rec, q, rep in jobs:
            if not rec.has(rep, q) and (rec.key, q, rep) not in seen:
                seen.add((rec.key, q, rep))
                todo.append((rec, q, rep))
        if not todo:
            return 0
        self.runner.reset_budget(None)
        self.runner.stage(label)
        cap = self.args.max_calls_per_question

        def one(job):
            rec, q, rep = job
            try:
                return M.run_program(rec.program, self.runner, self.rows[q], rep=rep, max_calls=cap)
            except M.OffCache:
                return None

        try:
            attempt = 0
            while True:
                attempt += 1
                pool = ThreadPoolExecutor(max_workers=self.args.workers)
                try:
                    results = list(tqdm(pool.map(one, todo), total=len(todo), unit="q", desc=label,
                                        leave=False, disable=len(todo) < 40))
                except BaseException:
                    # Ctrl-C or the call cap: drop what is queued, let the debates in
                    # flight stop at their next round (the budget is now 0), and only
                    # then return, so nothing writes to the cache after it is closed
                    self.runner.reset_budget(0)
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
                pool.shutdown()
                failed = []
                for job, o in zip(todo, results):
                    if o is None:
                        failed.append(job)
                    else:
                        job[0].record(job[2], job[1], o)
                todo = failed
                if not todo:
                    break
                if attempt < QUICK_ATTEMPTS:
                    continue
                if not getattr(self, "WAIT_FOR_SERVER", True):  # the caller completes or stops itself
                    log(f"  {label}: {len(todo)} debates could not be completed (server errors)")
                    break
                # Still failing after the quick tries (since 2026-10-06 a search never goes on with
                # debates missing: they were archived incomplete and never filled). A server that does
                # not answer is waited for; one that answers while debates fail (an engine down behind
                # it, a burst of server errors) gets a few slower tries. Then the search stops,
                # nonzero, and a rerun resumes from what was recorded.
                def stop(why: str):
                    raise SystemExit(f"{label}: {len(todo)} debates could not be completed after {attempt} "
                                     f"attempts ({why}; see the round cache's error lines). What was recorded "
                                     f"is kept: run the script again to resume.")
                if attempt >= MAX_ATTEMPTS:
                    stop("too many attempts")
                if server_up(self.args):
                    slow = attempt - QUICK_ATTEMPTS
                    if slow >= len(SLOW_WAITS):
                        stop("the server answers, so these debates fail on every try")
                    log(f"  {label}: {len(todo)} debates failed with the server up; "
                        f"trying them again in {SLOW_WAITS[slow]} s")
                    time.sleep(SLOW_WAITS[slow])
                    continue
                waited = 0
                while waited < OUTAGE_WAIT and not server_up(self.args):
                    if waited == 0:
                        log(f"  {label}: {len(todo)} debates failed and the server does not answer; "
                            f"waiting for it (up to {OUTAGE_WAIT // 60} min)")
                    time.sleep(60)
                    waited += 60
                if not server_up(self.args):
                    stop("the server did not come back")
                log(f"  {label}: the server answers again after {waited // 60} min; trying {len(todo)} debates again")
            return self.runner.novel_calls
        finally:
            self.runner.reset_budget(0)

    def score(self, rec: ProgRecord, g: int) -> float:
        return rec.score(self.gq[g])

    def turns(self, rec: ProgRecord, g: int) -> float:
        return rec.turns(self.gq[g]) or 0.0

    # -------------------------------------------------------------------- slots
    def compute_slots(self) -> dict[tuple, str]:
        """The 2k + 1 slot holders from the scores as they stand. Older programs
        win exact ties, so an equal newcomer never displaces a holder; with
        --tie-questions t, a newcomer up to t questions below the best score that
        uses fewer turns (at replicate 0) does take the slot (strongest)."""
        pool = self.eligible()
        slots: dict[tuple, str] = {}
        self.gaps_b = {}
        self.floor_setters = {}
        if not pool:
            return slots
        tie = tie_questions(self.args)
        best = strongest(pool, lambda r: r.score(self.qids), lambda r: r.turns(self.qids) or 0.0,
                         len(self.qids), tie)
        slots[(GLOBAL, GLOBAL_SLOT)] = best.key
        if tie > 0:     # the best score sets the tie floor: settle_slots runs it a second time too
            self.floor_setters[(GLOBAL, GLOBAL_SLOT)] = strongest(
                pool, lambda r: r.score(self.qids), lambda r: r.turns(self.qids) or 0.0, len(self.qids), 0).key
        pct = {g: percentile_ranks({r.key: self.score(r, g) for r in pool}) for g in self.group_ids}
        for g in self.group_ids:
            a = strongest(pool, lambda r: self.score(r, g), lambda r: self.turns(r, g), len(self.gq[g]), tie)
            slots[(g, "A")] = a.key
            if tie > 0:
                self.floor_setters[(g, "A")] = strongest(
                    pool, lambda r: self.score(r, g), lambda r: self.turns(r, g), len(self.gq[g]), 0).key
            others = [c for c in self.group_ids if c != g]
            if not others:
                continue
            for r in pool:
                self.gaps_b[(g, r.key)] = pct[g][r.key] - max(pct[c][r.key] for c in others)
            rest = [r for r in pool if r.key != a.key]
            if rest:
                b = min(rest, key=lambda r: (-round(self.gaps_b[(g, r.key)], 9), self.turns(r, g),
                                             r.gen, r.name))
                slots[(g, "B")] = b.key
        return slots

    def settle_slots(self) -> tuple[int, int]:
        """Work out the slots, give replicate 1 on the slot's group to any
        holder that lacks it, and repeat until every holder is confirmed.
        With --tie-questions, the program with the best score on a slot's
        questions sets that slot's tie floor, so it is run a second time too:
        a one-run score that is high by chance never sets the floor.
        Returns (speaker turns spent, programs confirmed)."""
        spent = n_confirmed = 0
        while True:
            slots = self.compute_slots()
            need = {(key, g) for (g, _), key in [*slots.items(), *self.floor_setters.items()]
                    if self.archive[key].gaps(self.gq[g], 1)}
            if not need:
                break
            before = sum(len(self.archive[key].gaps(self.gq[g], 1)) for key, g in need)
            spent += self.run_jobs([(self.archive[key], q, 1)
                                    for key, g in sorted(need, key=lambda t: (t[0], str(t[1])))
                                    for q in self.gq[g]], f"gen {self.gen} confirm {len(need)}")
            for key in {key for key, _ in need}:
                self.save(self.archive[key])
            n_confirmed += len(need)
            after = sum(len(self.archive[key].gaps(self.gq[g], 1)) for key, g in need)
            if after >= before:                              # server errors: do not spin
                log("  confirmation made no progress; keeping the provisional slots")
                break
        self.slots = slots
        return spent, n_confirmed

    # ----------------------------------------------------------------- children
    def breeding_form(self, rec: ProgRecord) -> tuple[dict, list[int]]:
        """`rec`'s program without the rules that never decided a step, and the
        kept rules' weights (steps decided, plus one)."""
        return P.prune(rec.program, rec.rule_fires())

    def draw_child(self, parent: ProgRecord, g, kind: str, name: str, siblings: list[ProgRecord],
                   taken: set[str], rng: random.Random, stats: Counter,
                   donors: list[dict] = ()) -> ProgRecord | None:
        """One non-neutral random edit of `parent` (pruned), replayed but not yet run."""
        gq = self.gq[g]
        same_as = {parent.behaviour(gq)} | {s.behaviour(gq) for s in siblings}
        same_as.discard(None)
        base, weights = self.breeding_form(parent)
        if len(base["rules"]) < len(parent.program["rules"]):
            stats["pruned_rules"] += len(parent.program["rules"]) - len(base["rules"])
        for attempt in range(MAX_REDRAWS):
            prog, op = P.mutate_by_family(base, rng, weights=weights, donors=donors)
            if op == "none":
                stats["redraw_no_edit"] += 1
                continue
            if P.plan_cost(prog) > P.MAX_TURNS:              # its last plan round could never run
                stats["redraw_over_cap"] += 1
                continue
            if P.reviewer_first(prog):                       # a reviewer first, with nothing to review
                stats["redraw_reviewer_first"] += 1
                continue
            key = P.canon(prog)
            if key in taken or key in self.archive or (key, g) in self.neutral:
                stats["redraw_known_text"] += 1
                continue
            rec = ProgRecord(prog, name, parent.lineage, self.gen, parent=parent.key, op=op)
            self.replay(rec, gq)
            if rec.behaviour(gq) in same_as:                  # None (not fully cached) is never in it
                self.neutral.add((key, g))
                stats["redraw_same_on_group"] += 1
                continue
            self.replay(rec, self.qids)
            h = rec.behaviour(self.qids)
            if h is not None and h in self.by_behaviour:
                self.neutral.add((key, g))
                stats["redraw_same_as_archived"] += 1
                continue
            taken.add(key)
            rec.meta = {"slot": kind, "target": g, "redraws": attempt}
            return rec
        stats["no_child"] += 1
        return None

    def unique_name(self, name: str, used: set[str]) -> str:
        while name in used:
            name += "r"
        used.add(name)
        return name

    # ------------------------------------------------------------- the loop
    def seed(self, seeds: list[dict]) -> None:
        """Generation 0: every seed on every search question, both replicates."""
        recs = []
        for s in seeds:
            prog = P.normalize_program(s["program"])
            P.validate_program(prog)
            key = P.canon(prog)
            if key in self.archive or key in {r.key for r in recs}:
                continue
            if (why := P.never_runs_as_written(prog)) is not None:
                raise SystemExit(f"seed {s['name']}: {why}")
            if P.reviewer_first(prog):
                raise SystemExit(f"seed {s['name']}: a critic, verifier or synthesizer speaks in its first "
                                 "round, where there is nothing to review")
            rec = self.archive.get(key) or ProgRecord(prog, s["name"], s["name"], 0,
                                                     op=s.get("source", "seed"))
            self.seed_source.setdefault(s["name"], s.get("source", "seed"))
            recs.append(rec)
        t0 = time.perf_counter()
        for rec in recs:
            for rep in (0, 1):
                self.replay(rec, self.qids, rep)
        spent = 0
        for rep in (0, 1):
            spent += self.run_jobs([(rec, q, rep) for rec in recs for q in self.qids],
                                   f"seeds replicate {rep}")
        for rec in recs:
            self.add(rec)
            log(f"  seed {rec.name:34s} {self.source_of(rec):8s} acc {rec.score(self.qids):.3f}  "
                f"turns {rec.turns(self.qids) or 0:.1f}")
        if recs:
            c_spent, _ = self.settle_slots()
            self.write_gen_log({"gen": 0, "seeds": len(recs), "turns_seeds": spent,
                                "turns_confirm": c_spent, "slots": self.slot_table(),
                                "live_calls": getattr(self.runner, "calls", 0),
                                "seconds": round(time.perf_counter() - t0, 1)})

    def generation(self) -> None:
        self.gen += 1
        t0 = time.perf_counter()
        # the draws depend on the seed, the generation and the archive (a generation stopped part way
        # is drawn again from its start, without the children it had made: load_archive)
        rng = random.Random(f"{self.args.seed}:{self.gen}:{len(self.archive)}")
        stats: Counter = Counter()
        before = dict(self.slots)
        parents = [(g, kind, self.archive[self.slots[(g, kind)]]) for g in self.group_ids
                   for kind in SLOT_KINDS if (g, kind) in self.slots]
        if (GLOBAL, GLOBAL_SLOT) in self.slots:
            parents.append((GLOBAL, GLOBAL_SLOT, self.archive[self.slots[(GLOBAL, GLOBAL_SLOT)]]))
        # what a crossover may take from: the other holders, pruned
        forms = {r.key: self.breeding_form(r)[0] for _, _, r in parents}
        used_names = {r.name for r in self.archive.values()}
        taken: set[str] = set()
        made: list[tuple[ProgRecord, ProgRecord, int]] = []
        spent = 0
        for wave in range(CHILDREN_PER_SLOT):       # a slot's second child is drawn after its first has run
            self.runner.stage(f"gen {self.gen} wave {wave + 1}: drawing children")
            batch: list[tuple[ProgRecord, ProgRecord, int]] = []
            for g, kind, parent in parents:
                sibs = [c for c, p, cg in made if p.key == parent.key and cg == g
                        and c.meta.get("slot") == kind]
                name = self.unique_name(f"g{self.gen}_{g}{kind}{wave + 1}", used_names)
                donors = [f for k, f in forms.items() if k != parent.key]
                child = self.draw_child(parent, g, kind, name, sibs, taken, rng, stats, donors)
                if child is not None:
                    batch.append((child, parent, g))
            spent += self.run_jobs([(c, q, 0) for c, _, _ in batch for q in self.qids],
                                   f"gen {self.gen} wave {wave + 1} ({len(batch)} children)")
            for child, _, _ in batch:
                stats[f"op:{child.op}"] += 1
                if (cut := P.cap_cuts(child, self.qids, 0)):     # the turn cap stopped some of its debates
                    stats["children_cut_by_cap"] += 1
                    stats["debates_cut_by_cap"] += cut
                self.add(child)
            made += batch
        spent_c, n_confirmed = self.settle_slots()
        new = {s: k for s, k in self.slots.items() if before.get(s) != k}
        for (g, kind), key in new.items():
            r = self.archive[key]
            stats[f"new_holder_{kind}"] += 1
            if r.gen == self.gen:
                stats[f"op_holder:{r.op}"] += 1
        entry = {"gen": self.gen, "children": len(made), "parents": len(parents),
                 "new_holders": len(new), "confirmed": n_confirmed,
                 "turns_children": spent, "turns_confirm": spent_c,
                 "archive": len(self.archive), "eligible": len(self.eligible()),
                 "live_calls": getattr(self.runner, "calls", 0),
                 "summaries": getattr(self.runner, "summaries", 0),
                 "errors": getattr(self.runner, "errors", 0), "v2": dict(D.V2_STATS),
                 "holder_sources": Counter(self.source_of(self.archive[k]) for k in self.slots.values()),
                 "slots": self.slot_table(), "seconds": round(time.perf_counter() - t0, 1), **stats}
        if D.JUDGE_ON:                   # judge calls so far: repicked = asked again to choose a candidate
            entry["judge"] = dict(D.JUDGE_STATS)
        self.write_gen_log(entry)
        redraws = sum(v for k, v in stats.items() if k.startswith("redraw_"))
        log(f"gen {self.gen:>3}: {len(made)} children ({redraws} redraws), new holders {len(new)}, "
            f"confirmed {n_confirmed}, "
            f"archive {len(self.archive)}, live calls {entry['live_calls']}, {entry['seconds']}s"
            + (f"; the turn cap stopped {stats['debates_cut_by_cap']} debates of {stats['children_cut_by_cap']} "
               f"children" if stats["children_cut_by_cap"] else ""))
        if self.gen % self.args.print_every == 0:
            self.print_slots()

    def ensure_seeds(self, seeds: list[dict]) -> None:
        """On resume: finish any seed an interrupted generation 0 left short."""
        short = [s for s in seeds
                 if (r := self.archive.get(P.canon(P.normalize_program(s["program"])))) is None
                 or r.gaps(self.qids, 0) or r.gaps(self.qids, 1)]
        if short:
            log(f"completing {len(short)} seeds")
            for s in short:
                self.archive.pop(P.canon(P.normalize_program(s["program"])), None)
            self.seed(short)

    # ------------------------------------------------------------- reporting
    def slot_table(self) -> list[dict]:
        out = []
        for g, kinds in [(g, SLOT_KINDS) for g in self.group_ids] + [(GLOBAL, (GLOBAL_SLOT,))]:
            for kind in kinds:
                key = self.slots.get((g, kind))
                if key is None:
                    continue
                r = self.archive[key]
                out.append({"group": g, "slot": kind, "name": r.name, "lineage": r.lineage,
                            "source": self.source_of(r), "gen": r.gen, "key": key,
                            "score": round(self.score(r, g), 4), "turns": round(self.turns(r, g), 2),
                            "rank_gap": round(self.gaps_b.get((g, key), 0.0), 4),
                            "overall": round(r.score(self.qids), 4),
                            "tokens": r.tokens_used(self.gq[g]),
                            "cut_by_cap": P.cap_cuts(r, self.gq[g], 0)})
        return out

    def print_slots(self) -> None:
        log(f"  {'group':>5} {'slot':>4} {'acc':>6} {'turns':>6} {'gap':>6} {'overall':>7} {'cut':>4}  program")
        for s in self.slot_table():
            log(f"  {s['group']:>5} {s['slot']:>4} {s['score']:>6.3f} {s['turns']:>6.1f} "
                f"{s['rank_gap']:>6.2f} {s['overall']:>7.3f} {s['cut_by_cap']:>4}  {s['name']} "
                f"(from {s['lineage']}, {s['source']})")
        log("  (cut: the slot's questions on which the turn cap stopped the holder's debate, replicate 0)")

    def write_gen_log(self, entry: dict) -> None:
        end_line(self.log_path)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(entry, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
                     + "\n")

    def summary(self) -> dict:
        slots = self.slot_table()
        for s in slots:
            s["program"] = self.archive[s["key"]].program
            s["program_pruned"] = self.breeding_form(self.archive[s["key"]])[0]
        pool = self.eligible()
        top = sorted(pool, key=lambda r: (-r.score(self.qids), r.turns(self.qids) or 0.0))[:5]
        by_source = Counter(self.source_of(r) for r in self.archive.values())
        return {"version": 3, "generations": self.gen, "archive": len(self.archive),
                "eligible": len(pool),
                "duplicates": sum(1 for r in self.archive.values() if r.dup_of is not None),
                "slots": slots, "archive_by_source": dict(by_source),
                "holders_by_source": dict(Counter(s["source"] for s in slots)),
                "top_overall": [{"score": r.score(self.qids), "turns": r.turns(self.qids), "name": r.name,
                                 "lineage": r.lineage, "source": self.source_of(r),
                                 "program": r.program} for r in top],
                "live_calls": getattr(self.runner, "calls", 0)}

    # ------------------------------------------------------------ champions
    def pick_champions(self) -> dict:
        """Per group: five finalists (the two slot holders, then the next
        strongest, distinct in behaviour) and the literature baselines run on a
        random sample of the group's held-out questions at two replicates. The
        champion is the finalist with the best held-out mean, ties to fewer
        turns (with --tie-questions t, every finalist within t held-out questions
        of the best mean ties, as in the slots). Also named: the cheapest of those programs within one paired
        standard error of the champion. The same over all search questions and
        the union of the held-out samples gives a global champion.

        The debates of every group run in one thread pool (since 2026-10-09; one
        pool per group before, one after another), so the slowest debates of the
        groups overlap instead of following each other. The groups' held-out
        questions are disjoint, so no group's debate can read another's
        recordings, and the same debates run as before. The global set runs
        after them, as before: a debate the groups recorded (a program that is a
        finalist of its group and of the global set) is read from the cache.
        The champions are then decided set by set."""
        self.slots = self.compute_slots()
        hrng = random.Random(self.args.seed + 7)
        held: dict[int, list[str]] = {}
        for g in self.groups["groups"]:
            pool = [q for q in g["held_out"] if q in self.rows]
            held[g["group"]] = sorted(hrng.sample(pool, min(len(pool), self.args.heldout_cap)))
        reps = list(range(self.args.reps))

        baselines: list[ProgRecord] = []
        for name in [b for b in self.args.baselines.split(",") if b]:
            prog = P.normalize_program(P.PROTOCOLS[name])
            # always under its literature name: a child can have the same text (the results are the
            # same, every program is replayed from the cache here), but the report names the baseline
            baselines.append(ProgRecord(prog, name, name, 0, op="protocol"))

        def finalists(g: int | None) -> list[ProgRecord]:
            qids = self.qids if g is None else self.gq[g]
            if g is None:
                firsts = [self.archive[self.slots[(GLOBAL, GLOBAL_SLOT)]]] if (GLOBAL, GLOBAL_SLOT) in self.slots else []
            else:
                firsts = [self.archive[self.slots[(g, k)]] for k in SLOT_KINDS if (g, k) in self.slots]
            ranked = sorted(self.eligible(), key=lambda r: (-r.score(qids), r.turns(qids) or 0.0,
                                                            r.gen, r.name))
            picked, seen_beh = [], set()
            for r in firsts + ranked:
                h = r.behaviour(qids)
                if r.key in {p.key for p in picked} or h in seen_beh:
                    continue
                picked.append(r)
                seen_beh.add(h)
                if len(picked) == self.args.top_n:
                    break
            return picked

        def fill(shadow: dict[str, ProgRecord], qids: list[str]) -> None:
            for s in shadow.values():
                for rep in reps:
                    self.replay(s, qids, rep)

        def prepare(g: int | None, qids: list[str]):
            """A set's finalists, its baselines that are not finalists, and a fresh record of each
            (held-out results are kept apart from the search record), filled from the cache."""
            fin = finalists(g)
            extra = [b for b in baselines if b.key not in {r.key for r in fin}]
            shadow = {r.key: ProgRecord(r.program, r.name, r.lineage, r.gen) for r in fin + extra}
            fill(shadow, qids)
            return fin, extra, shadow

        def evaluate(recs: list[ProgRecord], shadow: dict[str, ProgRecord], qids: list[str]) -> dict[str, dict]:
            out = {}
            for r in recs:
                s = shadow[r.key]
                accs = [sum(s.reps.get(rep, {}).get(q, [0])[0] for q in qids) / max(len(qids), 1)
                        for rep in reps]
                out[r.key] = {"name": r.name, "lineage": r.lineage, "source": self.source_of(r),
                              "key": r.key, "program": r.program,
                              "held_out_acc": sum(accs) / len(accs), "per_rep_acc": accs,
                              "turns": s.turns(qids), "tokens": s.tokens_used(qids),
                              "n_missing": sum(len(s.gaps(qids, rep)) for rep in reps),   # every replicate
                              "marks": {q: s.mark(q) or 0.0 for q in qids}}
            return out

        def decide(g: int | None, qids: list[str], search_q: list[str], fin: list[ProgRecord],
                   extra: list[ProgRecord], shadow: dict[str, ProgRecord]) -> dict:
            res = evaluate(fin + extra, shadow, qids)
            rows_f = []
            for r in fin + extra:
                d = res[r.key]
                d["finalist"] = r in fin
                d["baseline"] = r.key in {b.key for b in baselines}
                d["search_score"] = r.score(search_q) if not r.gaps(search_q, 0) else None
                rows_f.append(d)
            fin_rows = [d for d in rows_f if d["finalist"]]
            tie = tie_questions(self.args)
            if tie <= 0 or not fin_rows:
                order = sorted(fin_rows, key=lambda d: (-round(d["held_out_acc"], 9), d["turns"] or 0.0))
                champ = order[0] if order else None
            else:                    # the slot rule: within `tie` held-out questions of the best, fewest turns
                floor = round(max(d["held_out_acc"] for d in fin_rows) - tie / len(qids), 9)
                champ = min((d for d in fin_rows if round(d["held_out_acc"], 9) >= floor),
                            key=lambda d: (d["turns"] or 0.0, -round(d["held_out_acc"], 9)))
            cheapest = None
            if champ is not None:
                for d in rows_f:
                    mean, se = paired_se([champ["marks"][q] for q in qids], [d["marks"][q] for q in qids])
                    d["vs_champion"] = {"diff": round(mean, 4), "se": round(se, 4),
                                        "within_one_se": bool(mean <= se + EPS)}
                near = [d for d in rows_f if d["vs_champion"]["within_one_se"]]
                cheapest = min(near, key=lambda d: (d["turns"] or 0.0, -d["held_out_acc"]))
            rows_f.sort(key=lambda d: (-d["held_out_acc"], d["turns"] or 0.0))
            log(f"{'all groups' if g is None else 'group ' + str(g)}: held-out {len(qids)} q; " + "; ".join(
                f"{d['name']}{'*' if d['baseline'] else ''} {d['held_out_acc']:.3f}@{(d['turns'] or 0):.1f}"
                for d in rows_f))
            if champ is not None:
                log(f"    champion {champ['name']} {champ['held_out_acc']:.3f}; cheapest within one SE: "
                    f"{cheapest['name']} {cheapest['held_out_acc']:.3f}@{(cheapest['turns'] or 0):.1f}")
            return {"held_out_n": len(qids), "programs": rows_f,
                    "champion": champ["key"] if champ else None,
                    "cheapest_within_one_se": cheapest["key"] if cheapest else None}

        result: dict = {"version": 3, "heldout_cap": self.args.heldout_cap, "reps": self.args.reps,
                        "baselines": [b.name for b in baselines], "per_group": {}}
        if tie_questions(self.args):                    # named only when used, as in the archive header
            result["tie_questions"] = tie_questions(self.args)
        all_held = sorted({q for qs in held.values() for q in qs})
        sets = [(g, held[g], self.gq[g]) for g in self.group_ids] + [(None, all_held, self.qids)]
        prepared = []
        for part, label in ((sets[:-1], f"champions: {len(self.group_ids)} groups"), (sets[-1:], "champions all")):
            done = [prepare(g, qids) for g, qids, _ in part]
            # each group in the order it had alone; run_jobs runs a (program, question, replicate)
            # asked for twice once, and the fill gives it to the other record from the cache
            self.run_jobs([(s, q, rep) for (_, qids, _), (_, _, shadow) in zip(part, done)
                           for s in shadow.values() for rep in reps for q in qids], label)
            for (_, qids, _), (_, _, shadow) in zip(part, done):
                fill(shadow, qids)
            prepared += done
        for (g, qids, search_q), (fin, extra, shadow) in zip(sets, prepared):
            res = decide(g, qids, search_q, fin, extra, shadow)
            if g is None:
                result["global"] = res
            else:
                result["per_group"][str(g)] = res
        return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="run directory (archive, logs, results)")
    ap.add_argument("--seeds", type=Path, default=None, help="default: seeds.json beside the run directory")
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs/describe_v3/clusters_600_train.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_600_train.json")
    ap.add_argument("--per-group", type=int, default=50,
                    help="search questions per group, the rest held out for --pick-champions; "
                         "0 = every question of every group (nothing held out)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dev-split", type=Path, default=None,
                    help="a split file of split_train_dev.py: each group's dev questions are held out for "
                         "--pick-champions, the rest are search questions (with --per-group N > 0, the "
                         "first N of them)")
    ap.add_argument("--pick-champions", action="store_true", help="no search: champion picking only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--tie-questions", type=int, default=0,
                    help="slots A and G, and the champion step: every program within this many questions of the "
                         "best score (of the slot's questions) ties with it, and the fewest turns wins; 0 (the "
                         "default): only an equal score ties. Recorded in the archive header")
    ap.add_argument("--max-calls-per-question", type=int, default=None,
                    help="default: the turn cap (--turn-cap, 16 unless set)")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=5, help="finalists per group at champion picking")
    ap.add_argument("--reps", type=int, default=2, help="replicates for champion picking")
    ap.add_argument("--heldout-cap", type=int, default=150, help="held-out questions per group used for picking")
    ap.add_argument("--baselines", default="direct,mad,self_refine",
                    help="literature programs scored on the held-out questions beside the finalists")
    live = ap.add_argument_group("debate model")
    live.add_argument("--base-urls", default=P.DEFAULT_BASE_URLS)
    live.add_argument("--model", default=P.DEFAULT_MODEL)
    live.add_argument("--api-key", default="EMPTY")
    live.add_argument("--temperature", type=float, default=0.7)
    live.add_argument("--workers", type=int, default=64,
                      help="debates in flight at once; each may have up to 4 speakers in flight")
    live.add_argument("--live-cache", type=Path, default=None,
                      help="round cache (default rounds_<model>.jsonl beside the run directory)")
    live.add_argument("--max-total-calls", type=int, default=800_000,
                      help="a safety rail on model calls in this process, not a stopping rule")
    live.add_argument("--ignore-cache-lock", action="store_true")
    P.add_executor_args(ap)
    args = ap.parse_args()

    settings = P.configure_from_args(args)
    settings["model"] = args.model
    for b in [b for b in args.baselines.split(",") if b]:
        if b not in P.PROTOCOLS:
            raise SystemExit(f"--baselines: {b!r} is not one of {list(P.PROTOCOLS)}")
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    P.check_rows(rows.values())             # the dataset suits the answer mode
    groups = P.load_groups(args.clusters, args.per_group, args.dev_split)
    if args.seeds is None:
        args.seeds = Path(args.out).parent / "seeds.json"
    if args.live_cache is None:
        args.live_cache = Path(args.out).parent / f"rounds_{P.model_tag(args.model)}.jsonl"
    seeds = json.loads(args.seeds.read_text()) if args.seeds.exists() else None
    if seeds is None and not (args.resume or args.pick_champions):
        raise SystemExit(f"{args.seeds} does not exist; run program_seeds_cluster.py first")
    seed_source = {s["name"]: s.get("source", "seed") for s in seeds["seeds"]} if seeds else {}
    Path(args.live_cache).parent.mkdir(parents=True, exist_ok=True)
    runner = P.make_runner(rows, args.live_cache, args.base_urls, args.model, args.temperature,
                           max_total_calls=args.max_total_calls, api_key=args.api_key,
                           lock=not args.ignore_cache_lock)
    k = len(groups["groups"])
    log(f"executor {settings}; model {args.model}; cache {args.live_cache}")
    n_slots = len(SLOT_KINDS) * k + 1
    log(f"{k} groups, " + " + ".join(str(len(g['search'])) for g in groups['groups'])
        + f" = {sum(len(g['search']) for g in groups['groups'])} search questions; {n_slots} slots (A and B per "
        f"group, one global), {CHILDREN_PER_SLOT * n_slots} children per generation, {args.generations} "
        f"generations; held out " + ", ".join(str(len(g['held_out'])) for g in groups['groups']))
    search = Search(args, rows, groups, runner, settings, seed_source)
    opened = False
    try:
        if args.pick_champions:
            if not any(g["held_out"] for g in groups["groups"]):
                raise SystemExit("--pick-champions needs held-out questions, and every question is a search "
                                 "question (--per-group 0); evaluate the slot holders instead "
                                 "(eval_routed_dev.py --no-champions)")
            search.open_archive(resume=True)
            result = search.pick_champions()
            out = Path(args.out) / "champions.json"
            out.write_text(json.dumps(result, indent=1))
            log(f"champions -> {out}")
            return
        search.open_archive(resume=args.resume)
        opened = True
        if args.resume:
            if seeds is not None:
                search.ensure_seeds(seeds["seeds"])
            search.settle_slots()
        else:
            if seeds.get("settings", {}).get("model") not in (None, args.model):
                log(f"note: seeds were sanity-run on {seeds['settings'].get('model')}, this run uses {args.model}")
            log(f"generation 0: {len(seeds['seeds'])} seeds on {len(search.qids)} questions, two replicates")
            search.seed(seeds["seeds"])
        search.print_slots()
        while search.gen < args.generations:
            search.generation()
    except SF.BudgetExhausted as exc:
        log(f"\nstopped: {exc}")
    except KeyboardInterrupt:
        log("\ninterrupted; the archive on disk is complete up to the last finished wave")
    finally:
        try:
            if opened:
                search.slots = search.compute_slots()
                (Path(args.out) / "summary.json").write_text(json.dumps(search.summary(), indent=1))
                log(f"summary -> {Path(args.out) / 'summary.json'}")
                search.print_slots()
            if P.JUDGE is not None:
                log(f"judge {P.JUDGE.model}: {P.JUDGE.stats}")
        finally:
            runner.close()


if __name__ == "__main__":
    main()

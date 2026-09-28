"""Per-group evolutionary search for debate-control programs, v3.

One search over the k question groups of outputs/clusters_train_both_v3.json
(50 search questions per group). Every count is tied to k; nothing is split
by hand-set shares.

Slots. Each group keeps two programs, so there are 2k parents:
    A  strongest    the highest score on the group; ties go to fewer turns.
    B  specialist   the largest rank gap. Scores are turned into percentile
                    ranks inside each group; a program's gap on a group is its
                    rank there minus its highest rank on any other group.
                    Largest gap wins, ties go to fewer turns, and the group's
                    slot-A holder is skipped.
Only programs scored on every search question can hold a slot. The archive
only grows: a child never replaces its parent, it competes with it.

Confirmation. Every program is first scored once (replicate 0). A program
that would take a slot is first scored a second time (replicate 1) on that
group's questions; its score there becomes the two-replicate mean and the
slots are worked out again, until every holder has both replicates. A
program's score on a group is always the mean over the replicates it has.

Children. Each slot's holder has two children per generation (4k in all).
A child is one random edit whose kind is drawn uniformly from the kinds that
apply. It is first replayed against the recordings, which is free, and drawn
again if its text is already known, if it behaves exactly like its parent or
its sibling on the slot's group, or exactly like an archived program overall.

Screen. A child is scored on its slot's group first. It goes on to the other
groups only if it is within one paired standard error of its parent there:
    d[q] = parent's mark - child's mark (replicate 0, same questions)
    pass if mean(d) <= stdev(d) / sqrt(n)
Replicate 0 is used for both, because parent and child share recordings up to
the edit, so the difference isolates the edit. A child that fails is not run
on the other groups; it stays in the archive as a partial program and cannot
hold a slot (unless the recordings already cover it everywhere, which costs
nothing and makes it a fully scored program like any other).

Stop. A fixed number of generations (20): 20 x 4k new programs.

    python scripts/evolve_program_clusters_v3.py \\
        --seeds outputs/cluster_search_v3/seeds.json --out outputs/cluster_search_v3/run1
    python scripts/evolve_program_clusters_v3.py --out outputs/cluster_search_v3/run1 --resume

    # champions on held-out questions (two replicates), with the literature
    # baselines scored on the same questions
    python scripts/evolve_program_clusters_v3.py --out outputs/cluster_search_v3/run1 --pick-champions
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
SLOT_KINDS = ("A", "B")            # strongest, specialist
CHILDREN_PER_SLOT = 2
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


class Search:
    def __init__(self, args, rows: dict, groups: dict, runner, settings: dict,
                 seed_source: dict[str, str] | None = None):
        self.args, self.rows, self.groups, self.runner = args, rows, groups, runner
        self.settings = settings
        self.gq: dict[int, list[str]] = {g["group"]: list(g["search"]) for g in groups["groups"]}
        self.group_ids = [g["group"] for g in groups["groups"]]
        self.qids: list[str] = [q for g in self.group_ids for q in self.gq[g]]
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
        return {"header": True, "version": 3, "settings": self.settings,
                "clusters": self.groups["source"], "per_group": self.groups["per_group"],
                "k": self.groups["k"], "qids": self.qids,
                "groups": {str(g): self.gq[g] for g in self.group_ids},
                "seed": self.args.seed, "seed_source": self.seed_source}

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
        for field in ("version", "settings", "clusters", "k", "per_group", "qids"):
            if header.get(field) != mine[field]:
                raise SystemExit(f"--resume: the archive's {field} differs from this run's "
                                 f"(a v3 search has a fixed question set)")
        if not self.seed_source:
            self.seed_source = dict(header.get("seed_source") or {})
        for d in recs.values():
            rec = ProgRecord.from_json(d)
            rec.dup_of = None
            self.archive[rec.key] = rec
        for rec in self.archive.values():                   # file order = generation order
            self.mark_duplicate(rec)
        # the last COMPLETE generation is the last one logged; programs of an
        # interrupted generation stay in the archive and that generation is redone
        if self.log_path.exists():
            for line in self.log_path.open():
                try:
                    self.gen = max(self.gen, int(json.loads(line).get("gen", 0)))
                except (ValueError, TypeError):
                    pass
        self._archive_fh = self.archive_path.open("a")
        log(f"resumed {len(self.archive)} programs from {self.archive_path} "
            f"(last complete generation {self.gen})")

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
        again, twice at most. Returns the speaker turns spent."""
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
            for _ in range(3):
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
            if todo:
                log(f"  {label}: {len(todo)} debates could not be completed (server errors)")
            return self.runner.novel_calls
        finally:
            self.runner.reset_budget(0)

    def score(self, rec: ProgRecord, g: int) -> float:
        return rec.score(self.gq[g])

    def turns(self, rec: ProgRecord, g: int) -> float:
        return rec.turns(self.gq[g]) or 0.0

    # -------------------------------------------------------------------- slots
    def compute_slots(self) -> dict[tuple[int, str], str]:
        """The 2k slot holders from the scores as they stand. Older programs
        win exact ties, so an equal newcomer never displaces a holder."""
        pool = self.eligible()
        slots: dict[tuple[int, str], str] = {}
        self.gaps_b = {}
        if not pool:
            return slots
        pct = {g: percentile_ranks({r.key: self.score(r, g) for r in pool}) for g in self.group_ids}
        for g in self.group_ids:
            a = min(pool, key=lambda r: (-round(self.score(r, g), 9), self.turns(r, g), r.gen, r.name))
            slots[(g, "A")] = a.key
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
        Returns (speaker turns spent, programs confirmed)."""
        spent = n_confirmed = 0
        while True:
            slots = self.compute_slots()
            need = {(key, g) for (g, _), key in slots.items() if self.archive[key].gaps(self.gq[g], 1)}
            if not need:
                break
            before = sum(len(self.archive[key].gaps(self.gq[g], 1)) for key, g in need)
            spent += self.run_jobs([(self.archive[key], q, 1) for key, g in sorted(need)
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
    def screen(self, child: ProgRecord, parent: ProgRecord, g: int) -> tuple[bool, float, float]:
        """Within one paired standard error of the parent on group g, at
        replicate 0. Returns (pass, mean difference parent - child, SE)."""
        qs = [q for q in self.gq[g] if child.has(0, q) and parent.has(0, q)]
        if len(qs) < len(self.gq[g]):
            return False, float("nan"), float("nan")        # could not be scored on the whole group
        mean, se = paired_se([parent.reps[0][q][0] for q in qs], [child.reps[0][q][0] for q in qs])
        return mean <= se + EPS, mean, se

    def draw_child(self, parent: ProgRecord, g: int, kind: str, name: str, siblings: list[ProgRecord],
                   taken: set[str], rng: random.Random, stats: Counter) -> ProgRecord | None:
        """One non-neutral random edit of `parent`, replayed but not yet run."""
        gq = self.gq[g]
        same_as = {parent.behaviour(gq)} | {s.behaviour(gq) for s in siblings}
        same_as.discard(None)
        for attempt in range(MAX_REDRAWS):
            prog, op = P.mutate_uniform(parent.program, rng)
            if op == "none":
                stats["redraw_no_edit"] += 1
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
        # the draws depend on the seed, the generation and the archive, so a
        # generation redone after an interruption draws the same children
        rng = random.Random(f"{self.args.seed}:{self.gen}:{len(self.archive)}")
        stats: Counter = Counter()
        before = dict(self.slots)
        parents = [(g, kind, self.archive[self.slots[(g, kind)]]) for g in self.group_ids
                   for kind in SLOT_KINDS if (g, kind) in self.slots]
        used_names = {r.name for r in self.archive.values()}
        taken: set[str] = set()
        made: list[tuple[ProgRecord, ProgRecord, int]] = []
        spent1 = spent2 = 0
        for wave in range(CHILDREN_PER_SLOT):       # a slot's second child is drawn after its first has run
            self.runner.stage(f"gen {self.gen} wave {wave + 1}: drawing children")
            batch: list[tuple[ProgRecord, ProgRecord, int]] = []
            for g, kind, parent in parents:
                sibs = [c for c, p, cg in made if p.key == parent.key and cg == g
                        and c.meta.get("slot") == kind]
                name = self.unique_name(f"g{self.gen}_{g}{kind}{wave + 1}", used_names)
                child = self.draw_child(parent, g, kind, name, sibs, taken, rng, stats)
                if child is not None:
                    batch.append((child, parent, g))
            spent1 += self.run_jobs([(c, q, 0) for c, _, g in batch for q in self.gq[g]],
                                    f"gen {self.gen} wave {wave + 1} stage 1 ({len(batch)} children)")
            passed = []
            for child, parent, g in batch:
                ok, mean, se = self.screen(child, parent, g)
                child.meta.update({"screen": bool(ok), "d": None if math.isnan(mean) else round(mean, 4),
                                   "se": None if math.isnan(se) else round(se, 4)})
                stats["screen_pass" if ok else "screen_fail"] += 1
                stats[f"op:{child.op}"] += 1
                if ok:
                    stats[f"op_pass:{child.op}"] += 1
                    passed.append(child)
            spent2 += self.run_jobs([(c, q, 0) for c in passed for q in self.qids],
                                    f"gen {self.gen} wave {wave + 1} stage 2 ({len(passed)} children)")
            for child, _, _ in batch:
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
                 "turns_stage1": spent1, "turns_stage2": spent2, "turns_confirm": spent_c,
                 "archive": len(self.archive), "eligible": len(self.eligible()),
                 "live_calls": getattr(self.runner, "calls", 0),
                 "summaries": getattr(self.runner, "summaries", 0),
                 "errors": getattr(self.runner, "errors", 0), "v2": dict(D.V2_STATS),
                 "holder_sources": Counter(self.source_of(self.archive[k]) for k in self.slots.values()),
                 "slots": self.slot_table(), "seconds": round(time.perf_counter() - t0, 1), **stats}
        self.write_gen_log(entry)
        redraws = sum(v for k, v in stats.items() if k.startswith("redraw_"))
        log(f"gen {self.gen:>3}: {len(made)} children ({stats['screen_pass']} passed the screen, "
            f"{redraws} redraws), new holders {len(new)}, confirmed {n_confirmed}, "
            f"archive {len(self.archive)}, live calls {entry['live_calls']}, {entry['seconds']}s")
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
        for g in self.group_ids:
            for kind in SLOT_KINDS:
                key = self.slots.get((g, kind))
                if key is None:
                    continue
                r = self.archive[key]
                out.append({"group": g, "slot": kind, "name": r.name, "lineage": r.lineage,
                            "source": self.source_of(r), "gen": r.gen, "key": key,
                            "score": round(self.score(r, g), 4), "turns": round(self.turns(r, g), 2),
                            "rank_gap": round(self.gaps_b.get((g, key), 0.0), 4),
                            "overall": round(r.score(self.qids), 4)})
        return out

    def print_slots(self) -> None:
        log(f"  {'group':>5} {'slot':>4} {'acc':>6} {'turns':>6} {'gap':>6} {'overall':>7}  program")
        for s in self.slot_table():
            log(f"  {s['group']:>5} {s['slot']:>4} {s['score']:>6.3f} {s['turns']:>6.1f} "
                f"{s['rank_gap']:>6.2f} {s['overall']:>7.3f}  {s['name']} "
                f"(from {s['lineage']}, {s['source']})")

    def write_gen_log(self, entry: dict) -> None:
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(entry, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
                     + "\n")

    def summary(self) -> dict:
        slots = self.slot_table()
        for s in slots:
            s["program"] = self.archive[s["key"]].program
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
        turns. Also named: the cheapest of those programs within one paired
        standard error of the champion. The same over all search questions and
        the union of the held-out samples gives a global champion."""
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
            baselines.append(self.archive.get(P.canon(prog)) or ProgRecord(prog, name, name, 0, op="protocol"))

        def finalists(g: int | None) -> list[ProgRecord]:
            qids = self.qids if g is None else self.gq[g]
            firsts = [] if g is None else [self.archive[self.slots[(g, k)]] for k in SLOT_KINDS
                                           if (g, k) in self.slots]
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

        def evaluate(recs: list[ProgRecord], qids: list[str], label: str) -> dict[str, dict]:
            """Held-out results are kept apart from the search record."""
            shadow = {r.key: ProgRecord(r.program, r.name, r.lineage, r.gen) for r in recs}
            for s in shadow.values():
                for rep in reps:
                    self.replay(s, qids, rep)
            self.run_jobs([(s, q, rep) for s in shadow.values() for rep in reps for q in qids], label)
            out = {}
            for r in recs:
                s = shadow[r.key]
                accs = [sum(s.reps.get(rep, {}).get(q, [0])[0] for q in qids) / max(len(qids), 1)
                        for rep in reps]
                out[r.key] = {"name": r.name, "lineage": r.lineage, "source": self.source_of(r),
                              "key": r.key, "program": r.program,
                              "held_out_acc": sum(accs) / len(accs), "per_rep_acc": accs,
                              "turns": s.turns(qids), "n_missing": len(s.gaps(qids, 0)),
                              "marks": {q: s.mark(q) or 0.0 for q in qids}}
            return out

        def decide(g: int | None, qids: list[str], search_q: list[str]) -> dict:
            fin = finalists(g)
            extra = [b for b in baselines if b.key not in {r.key for r in fin}]
            res = evaluate(fin + extra, qids, f"champions {'all' if g is None else 'group ' + str(g)}")
            rows_f = []
            for r in fin + extra:
                d = res[r.key]
                d["finalist"] = r in fin
                d["baseline"] = r.key in {b.key for b in baselines}
                d["search_score"] = r.score(search_q) if not r.gaps(search_q, 0) else None
                rows_f.append(d)
            order = sorted((d for d in rows_f if d["finalist"]),
                           key=lambda d: (-round(d["held_out_acc"], 9), d["turns"] or 0.0))
            champ = order[0] if order else None
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
        for g in self.group_ids:
            result["per_group"][str(g)] = decide(g, held[g], self.gq[g])
        all_held = sorted({q for qs in held.values() for q in qs})
        result["global"] = decide(None, all_held, self.qids)
        return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="run directory (archive, logs, results)")
    ap.add_argument("--seeds", type=Path, default=None, help="default: seeds.json beside the run directory")
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs/clusters_train_both_v3.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_program_search_train.json")
    ap.add_argument("--per-group", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pick-champions", action="store_true", help="no search: champion picking only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--max-calls-per-question", type=int, default=16)
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
    ap.add_argument("--digest-head", type=int, default=P.DIGEST_HEAD)
    ap.add_argument("--digest-tail", type=int, default=P.DIGEST_TAIL)
    ap.add_argument("--visible-reasoning", action="store_true",
                    help="for models that think in a hidden channel (gpt-oss): ask every speaker to "
                         "write its reasoning in the visible reply; changes prompts and cache keys")
    P.add_executor_args(ap)
    args = ap.parse_args()

    settings = P.configure_executor(args.digest_head, args.digest_tail, args.visible_reasoning,
                                    deep_think=args.deep_think, summary_words=args.summary_words)
    settings["model"] = args.model
    for b in [b for b in args.baselines.split(",") if b]:
        if b not in P.PROTOCOLS:
            raise SystemExit(f"--baselines: {b!r} is not one of {list(P.PROTOCOLS)}")
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    groups = P.load_groups(args.clusters, args.per_group)
    if args.seeds is None:
        args.seeds = Path(args.out).parent / "seeds.json"
    if args.live_cache is None:
        args.live_cache = Path(args.out).parent / f"rounds_{P.model_tag(args.model)}.jsonl"
    seeds = json.loads(args.seeds.read_text()) if args.seeds.exists() else None
    if seeds is None and not (args.resume or args.pick_champions):
        raise SystemExit(f"{args.seeds} does not exist; run program_seeds_v3.py first")
    seed_source = {s["name"]: s.get("source", "seed") for s in seeds["seeds"]} if seeds else {}
    Path(args.live_cache).parent.mkdir(parents=True, exist_ok=True)
    runner = P.make_runner(rows, args.live_cache, args.base_urls, args.model, args.temperature,
                           max_total_calls=args.max_total_calls, api_key=args.api_key,
                           lock=not args.ignore_cache_lock)
    k = len(groups["groups"])
    log(f"executor: v2, digest {settings['digest']}, {settings['answer_tokens']}-token replies, "
        f"model {args.model}; cache {args.live_cache}")
    log(f"{k} groups x {args.per_group} = {sum(len(g['search']) for g in groups['groups'])} search "
        f"questions; {len(SLOT_KINDS) * k} slots, {len(SLOT_KINDS) * CHILDREN_PER_SLOT * k} children per "
        f"generation, {args.generations} generations; held out "
        + ", ".join(str(len(g['held_out'])) for g in groups['groups']))
    search = Search(args, rows, groups, runner, settings, seed_source)
    opened = False
    try:
        if args.pick_champions:
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
        finally:
            runner.close()


if __name__ == "__main__":
    main()

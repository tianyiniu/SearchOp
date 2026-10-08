"""Evolutionary search for debate-control programs: the search of the global pipeline
(run_pipeline_global.sh; named evolve_program_v4.py until 2026-10-06).

One search over every question of the train split (no question groups, no routing). The
rules of a program, checked after every round, are what treat questions differently.

Cost levels. A program's cost is its average speaker turns per train question (a
high-effort speaker counts --high-cost turns, a low-effort one 1). Each program belongs
to one level by that average:

    cheap       up to 4.5 turns
    medium      more than 4.5, up to 10.5 turns
    expensive   more than 10.5 turns (the turn cap bounds it)

Slots. Each level keeps one program: the best score on the train split among the level's
programs, ties to fewer turns, then to the older program. Only programs scored on every
train question can hold a slot, and a program whose behaviour equals an earlier
program's on every train question is a duplicate, never selected. The archive only grows.

Scoring. Every program is scored once (replicate 0) on every train question. There is
no second run: the final choice is made on the dev split instead (below).

Children. Each slot's holder has 5 children per generation, drawn one wave at a time (a
holder's next child is drawn after its earlier ones have run, so it can differ from
them): 15 children per generation with all three levels held. A child is one random
edit of the pruned holder (as in the cluster search: rule, plan, effort, visibility and width
edits, and crossover with another level's holder), replayed for free first and drawn
again if it is known or behaves like its parent, a sibling or an archived program. A
child joins the level of its own cost, whichever level its parent held.

Resume. A generation's parents and draws depend only on the seed, the generation and the
programs of earlier generations. So a generation redone after an interruption has the
same parents and draws the same children. A child is put on file when it is drawn, before
it runs, so a child of the interrupted try is taken as it is, whatever it ran.

Server errors. A debate that fails is tried again (evolve_pipeline_cluster.run_jobs).
A generation is logged only when every program has every train result: debates still
missing after one more pass stop the search (exit code 1) with the generation unlogged,
so a rerun after the server is back redoes it, and no generation breeds from a part of
the one before.

Final choice (--select-dev, after the last generation). For each level: the 2 programs
with the best train score, and the level's best seed if it is not one of them, are run
once on the dev split (questions the search never ran). The one with the best dev score
is the level's final program (ties: fewer dev turns, then the better train score, then
the older program). The dev results are kept apart from the archive, in final.json.
Report the test scores of the final programs, not their train or dev scores: both hold
the luck of a selection.

    python scripts/evolve_pipeline_global.py --splits <run>/splits.json --seeds <run>/seeds.json \\
        --out <run> --generations 10 [--resume]
    python scripts/evolve_pipeline_global.py --splits <run>/splits.json --out <run> --select-dev
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import debate_mcq as D  # noqa: E402
import evolve_pipeline_cluster as V3  # noqa: E402
import program_space as P  # noqa: E402
import schema_fitness as SF  # noqa: E402
from program_space import ProgRecord  # noqa: E402
from split_train_dev import load_split  # noqa: E402

ROOT = P.ROOT
LEVELS: tuple[tuple[str, float | None], ...] = (("cheap", 4.5), ("medium", 10.5), ("expensive", None))
SLOT = "best"                      # the one slot of each level
CHILDREN_PER_LEVEL = 5
TOP_ON_DEV = 2                     # programs per level, by train score, run on the dev split
log = V3.log


def level_of(turns: float) -> str:
    """The cost level of an average of `turns` speaker turns per question."""
    for name, top in LEVELS:
        if top is None or turns <= top + 1e-9:
            return name
    raise AssertionError("the last level has no upper limit")


def mend_last_line(path: Path) -> None:
    """End a last line torn by a hard stop mid-write, so the next record starts on a line of
    its own (readers skip the torn record)."""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as fh:
            fh.seek(-1, 2)
            torn = fh.read(1) != b"\n"
        if torn:
            with path.open("a") as fh:
                fh.write("\n")


def level_text(turn_cap: int) -> list[tuple[str, str]]:
    """(name, description) of each level, for the seed writer."""
    out, low = [], None
    for name, top in LEVELS:
        if low is None:
            desc = f"up to {top:g} turns"
        elif top is None:
            desc = f"more than {low:g} turns (a question is capped at {turn_cap})"
        else:
            desc = f"more than {low:g}, up to {top:g} turns"
        out.append((name, desc))
        low = top
    return out


class Search(V3.Search):
    """The cluster search's machinery (archive, replays, debates in one pool, child draws),
    with one train question set, three cost-level slots and no second run."""

    WAIT_FOR_SERVER = False        # debates still missing get one more pass, then the search stops (finish_or_stop)
    DROP_STOPPED_GENERATION = False   # an interrupted generation's children are taken as they are (its redo)

    def __init__(self, args, rows: dict, splits: dict, runner, settings: dict,
                 seed_source: dict[str, str] | None = None):
        groups = {"groups": [{"group": 0, "search": list(splits["train"])}],
                  "source": None, "per_group": None, "k": None}
        super().__init__(args, rows, groups, runner, settings, seed_source)
        self.dev: list[str] = list(splits["dev"])
        self.dev_set = set(self.dev)
        self.allow_dev = False                    # only select_on_dev may run dev questions

    # ------------------------------------------------------------------ archive
    def header(self) -> dict:
        # clusters, k and per_group are None: the v3 resume check compares them too
        return {"header": True, "version": 4, "settings": self.settings, "clusters": None, "k": None,
                "per_group": None, "qids": self.qids, "dev": self.dev,
                "levels": [[name, top] for name, top in LEVELS], "children_per_level": CHILDREN_PER_LEVEL,
                "seed": self.args.seed, "seed_source": self.seed_source}

    def load_archive(self) -> None:
        mend_last_line(self.archive_path)
        mend_last_line(self.log_path)
        if self.archive_path.exists():
            with self.archive_path.open() as fh:
                first = fh.readline()
            try:
                header = json.loads(first)
            except json.JSONDecodeError:
                header = {}
            mine = self.header()
            for field in ("version", "dev", "levels", "children_per_level", "seed"):
                if header.get(field) != mine[field]:
                    raise SystemExit(f"--resume: the archive's {field} differs from this run's")
        super().load_archive()                    # checks version, settings and the train questions

    # ------------------------------------------------------------------ scoring
    def run_jobs(self, jobs, label: str) -> int:
        if not self.allow_dev and any(q in self.dev_set for _, q, _ in jobs):
            raise RuntimeError("the search tried to run a dev question")
        return super().run_jobs(jobs, label)

    def level(self, rec: ProgRecord) -> str:
        return level_of(rec.turns(self.qids) or 0.0)

    @staticmethod
    def rank_key(rec: ProgRecord, qids: list[str]):
        """Best score first, then fewer turns, then the older program."""
        return (-round(rec.score(qids), 9), rec.turns(qids) or 0.0, rec.gen, rec.name)

    # -------------------------------------------------------------------- slots
    def compute_slots(self, before_gen: int | None = None) -> dict[tuple, str]:
        """The level slots over the eligible programs (with `before_gen`, only those of
        earlier generations)."""
        pool = [r for r in self.eligible() if before_gen is None or r.gen < before_gen]
        self.gaps_b = {}
        slots: dict[tuple, str] = {}
        for name, _ in LEVELS:
            members = [r for r in pool if self.level(r) == name]
            if members:
                slots[(name, SLOT)] = min(members, key=lambda r: self.rank_key(r, self.qids)).key
        return slots

    def settle_slots(self) -> tuple[int, int]:
        """No second run in the global search: the slots follow from the scores as they stand."""
        self.slots = self.compute_slots()
        return 0, 0

    # ------------------------------------------------------------- the loop
    def seed(self, seeds: list[dict]) -> None:
        """Generation 0: every seed on every train question, once. A seed already on file
        with results missing (an interrupted generation 0) is completed in place."""
        recs = []
        for s in seeds:
            prog = P.normalize_program(s["program"])
            P.validate_program(prog)
            key = P.canon(prog)
            if key in {r.key for r in recs}:
                continue
            if key not in self.archive and (why := P.never_runs_as_written(prog)) is not None:
                raise SystemExit(f"seed {s['name']}: {why}")
            rec = self.archive.get(key)
            if rec is None:
                rec = ProgRecord(prog, s["name"], s["name"], 0, op=s.get("source", "seed"))
                self.seed_source.setdefault(s["name"], s.get("source", "seed"))
            elif not rec.gaps(self.qids, 0):
                continue
            recs.append(rec)
        t0 = time.perf_counter()
        for rec in recs:
            self.replay(rec, self.qids, 0)
        spent = self.run_jobs([(rec, q, 0) for rec in recs for q in self.qids], "seeds")
        for rec in recs:
            self.add(rec)
        self.redo_duplicates()                    # a seed completed in place keeps its place in file order
        self.finish_or_stop(recs, "generation 0")
        for rec in recs:
            log(f"  seed {rec.name:34s} {self.source_of(rec):8s} acc {rec.score(self.qids):.3f}  "
                f"turns {rec.turns(self.qids) or 0:.1f}  level {self.level(rec)}")
        if recs:
            self.settle_slots()
            self.write_gen_log({"gen": 0, "seeds": len(recs), "new_calls_seeds": spent, "slots": self.slot_table(),
                                "live_calls": getattr(self.runner, "calls", 0),
                                "seconds": round(time.perf_counter() - t0, 1)})

    def ensure_seeds(self, seeds: list[dict]) -> None:
        """On resume: finish any seed an interrupted generation 0 left short."""
        short = [s for s in seeds
                 if (r := self.archive.get(P.canon(P.normalize_program(s["program"])))) is None
                 or r.gaps(self.qids, 0)]
        if short:
            log(f"completing {len(short)} seeds")
            self.seed(short)
        elif 0 not in self.logged_generations():
            # stopped after the last seed was saved and before generation 0 was logged
            self.settle_slots()
            self.write_gen_log({"gen": 0, "seeds": sum(r.gen == 0 for r in self.archive.values()),
                                "new_calls_seeds": 0, "slots": self.slot_table(), "logged_on_resume": True})

    def logged_generations(self) -> list[int]:
        """The generations in the log, in order (a generation is complete once logged)."""
        out = []
        if self.log_path.exists():
            for line in self.log_path.open():
                try:
                    out.append(int(json.loads(line)["gen"]))
                except (ValueError, TypeError, KeyError):
                    pass
        return out

    def complete_short(self, stats: Counter) -> None:
        """Finish the train debates of programs of earlier generations that server errors
        left short (there are none in a run without errors). A short program can neither hold
        a slot nor be drawn again, so without this it would be lost."""
        short = [r for r in self.archive.values() if r.gen < self.gen and r.gaps(self.qids, 0)]
        if not short:
            return
        log(f"gen {self.gen}: completing {len(short)} programs that server errors left short")
        self.run_jobs([(r, q, 0) for r in short for q in r.gaps(self.qids, 0)],
                      f"gen {self.gen}: completing {len(short)} programs")
        for r in short:
            self.save(r)
        self.redo_duplicates()
        stats["completed_short"] = sum(not r.gaps(self.qids, 0) for r in short)

    def redo_duplicates(self) -> None:
        """The duplicates worked out again in archive order, as a reload works them out (needed
        once a program that was short has every result)."""
        self.by_behaviour = {}
        for r in self.archive.values():
            r.dup_of = None
        for r in self.archive.values():
            self.mark_duplicate(r)

    def finish_or_stop(self, recs: list[ProgRecord], label: str) -> None:
        """Debates of `recs` that server errors left missing get one more pass. If any is still
        missing, the search stops here, before the generation is logged: the server is likely
        down, and a rerun redoes the generation once it is back."""
        short = [r for r in recs if r.gaps(self.qids, 0)]
        if not short:
            return
        log(f"{label}: {sum(len(r.gaps(self.qids, 0)) for r in short)} debates of {len(short)} programs "
            f"are missing after server errors; trying them once more")
        self.run_jobs([(r, q, 0) for r in short for q in r.gaps(self.qids, 0)], f"{label}: missing debates")
        for r in short:
            self.save(r)
        self.redo_duplicates()
        left = [r for r in short if r.gaps(self.qids, 0)]
        if left:
            raise SystemExit(f"{label}: {sum(len(r.gaps(self.qids, 0)) for r in left)} debates of {len(left)} "
                             f"programs could not be completed (server errors; is the server up?). The "
                             f"generation is not logged: run again to redo it.")

    def draw_child(self, parent: ProgRecord, g, kind: str, name: str, siblings: list[ProgRecord],
                   taken: set[str], rng: random.Random, stats: Counter, donors: list[dict] = (),
                   redo: dict[str, ProgRecord] | None = None) -> ProgRecord | None:
        """V3.Search.draw_child, plus: a drawn program that already ran as this parent's child in
        an interrupted try of this generation (`redo`) is that child again."""
        gq = self.gq[g]
        same_as = {parent.behaviour(gq)} | {s.behaviour(gq) for s in siblings}
        same_as.discard(None)
        base, weights = self.breeding_form(parent)
        if len(base["rules"]) < len(parent.program["rules"]):
            stats["pruned_rules"] += len(parent.program["rules"]) - len(base["rules"])
        for attempt in range(V3.MAX_REDRAWS):
            prog, op = P.mutate_uniform(base, rng, weights=weights, donors=donors)
            if op == "none":
                stats["redraw_no_edit"] += 1
                continue
            if P.plan_cost(prog) > P.MAX_TURNS:              # its last plan round could never run
                stats["redraw_over_cap"] += 1
                continue
            key = P.canon(prog)
            old = (redo or {}).get(key)
            if (old is not None and key not in taken and old.parent == parent.key
                    and old.meta.get("slot") == kind):
                taken.add(key)
                stats["redone_child"] += 1
                return old
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

    def generation(self) -> None:
        self.gen += 1
        t0 = time.perf_counter()
        stats: Counter = Counter()
        self.complete_short(stats)
        # The parents and the draws depend only on the seed, the generation and the programs of
        # earlier generations, so a generation redone after an interruption has the same parents
        # and draws the same children; the children that already ran are taken as they are.
        older = [r for r in self.archive.values() if r.gen < self.gen]
        redo = {r.key: r for r in self.archive.values() if r.gen == self.gen}
        if redo:
            log(f"gen {self.gen}: redoing an interrupted generation ({len(redo)} of its children had run)")
        rng = random.Random(f"{self.args.seed}:{self.gen}:{len(older)}")
        before = self.compute_slots(before_gen=self.gen)
        parents = [(name, self.archive[before[(name, SLOT)]]) for name, _ in LEVELS if (name, SLOT) in before]
        forms = {r.key: self.breeding_form(r)[0] for _, r in parents}      # what a crossover may take from
        used_names = {r.name for r in self.archive.values()}
        taken: set[str] = set()
        made: list[tuple[ProgRecord, ProgRecord, str]] = []
        spent = 0
        for wave in range(CHILDREN_PER_LEVEL):
            self.runner.stage(f"gen {self.gen} wave {wave + 1}: drawing children")
            batch: list[tuple[ProgRecord, ProgRecord, str]] = []
            for level, parent in parents:
                sibs = [c for c, p, _ in made if p.key == parent.key]
                name = self.unique_name(f"g{self.gen}_{level}{wave + 1}", used_names)
                donors = [f for k, f in forms.items() if k != parent.key]
                child = self.draw_child(parent, V3.GLOBAL, level, name, sibs, taken, rng, stats, donors, redo)
                if child is not None:
                    if child.key not in redo:
                        self.save(child)          # on file before it runs: a redo takes it as it is
                    batch.append((child, parent, level))
            spent += self.run_jobs([(c, q, 0) for c, _, _ in batch for q in self.qids],
                                   f"gen {self.gen} wave {wave + 1} ({len(batch)} children)")
            for child, _, _ in batch:
                stats[f"op:{child.op}"] += 1
                stats[f"child_level:{self.level(child)}" if not child.gaps(self.qids, 0) else "child_incomplete"] += 1
                self.add(child)
            made += batch
        # every program must have every train result before the generation counts
        self.finish_or_stop([r for r in self.archive.values() if r.gaps(self.qids, 0)], f"gen {self.gen}")
        self.settle_slots()
        new = {s: k for s, k in self.slots.items() if before.get(s) != k}
        for (level, _), key in new.items():
            stats[f"new_holder_{level}"] += 1
            if self.archive[key].gen == self.gen:
                stats[f"op_holder:{self.archive[key].op}"] += 1
        entry = {"gen": self.gen, "children": len(made), "parents": len(parents), "new_holders": len(new),
                 "new_calls_children": spent, "archive": len(self.archive), "eligible": len(self.eligible()),
                 "live_calls": getattr(self.runner, "calls", 0), "summaries": getattr(self.runner, "summaries", 0),
                 "errors": getattr(self.runner, "errors", 0), "v2": dict(D.V2_STATS),
                 "holder_sources": Counter(self.source_of(self.archive[k]) for k in self.slots.values()),
                 "slots": self.slot_table(), "seconds": round(time.perf_counter() - t0, 1), **stats}
        self.write_gen_log(entry)
        redraws = sum(v for k, v in stats.items() if k.startswith("redraw_"))
        log(f"gen {self.gen:>3}: {len(made)} children ({redraws} redraws), new holders {len(new)}, "
            f"archive {len(self.archive)}, live calls {entry['live_calls']}, {entry['seconds']}s")
        if self.gen % self.args.print_every == 0:
            self.print_slots()

    # ------------------------------------------------------------- reporting
    def slot_table(self) -> list[dict]:
        out = []
        for name, _ in LEVELS:
            key = self.slots.get((name, SLOT))
            if key is None:
                continue
            r = self.archive[key]
            out.append({"level": name, "name": r.name, "lineage": r.lineage, "source": self.source_of(r),
                        "gen": r.gen, "key": key, "score": round(r.score(self.qids), 4),
                        "turns": round(r.turns(self.qids) or 0.0, 2), "tokens": r.tokens_used(self.qids)})
        return out

    def print_slots(self) -> None:
        log(f"  {'level':>9} {'acc':>6} {'turns':>6}  program")
        for s in self.slot_table():
            log(f"  {s['level']:>9} {s['score']:>6.3f} {s['turns']:>6.1f}  {s['name']} "
                f"(from {s['lineage']}, {s['source']})")

    def summary(self) -> dict:
        slots = self.slot_table()
        for s in slots:
            s["program"] = self.archive[s["key"]].program
            s["program_pruned"] = self.breeding_form(self.archive[s["key"]])[0]
        pool = self.eligible()
        by_level = {name: sorted([r for r in pool if self.level(r) == name],
                                 key=lambda r: self.rank_key(r, self.qids)) for name, _ in LEVELS}
        return {"version": 4, "generations": max(self.logged_generations(), default=-1),
                "archive": len(self.archive), "eligible": len(pool),
                "duplicates": sum(1 for r in self.archive.values() if r.dup_of is not None),
                "slots": slots, "programs_by_level": {k: len(v) for k, v in by_level.items()},
                "top_by_level": {k: [{"score": r.score(self.qids), "turns": r.turns(self.qids), "name": r.name,
                                      "lineage": r.lineage, "source": self.source_of(r), "gen": r.gen}
                                     for r in v[:5]] for k, v in by_level.items()},
                "archive_by_source": dict(Counter(self.source_of(r) for r in self.archive.values())),
                "holders_by_source": dict(Counter(s["source"] for s in slots)),
                "live_calls": getattr(self.runner, "calls", 0)}

    # ---------------------------------------------------------- dev choice
    def dev_candidates(self) -> dict[str, list[ProgRecord]]:
        """Per level: the TOP_ON_DEV best programs on the train split, then the level's
        best seed if it is not one of them."""
        pool = self.eligible()
        out = {}
        for name, _ in LEVELS:
            members = sorted([r for r in pool if self.level(r) == name], key=lambda r: self.rank_key(r, self.qids))
            cands = members[:TOP_ON_DEV]
            seed = next((r for r in members if r.gen == 0), None)
            if seed is not None and seed.key not in {r.key for r in cands}:
                cands.append(seed)
            out[name] = cands
        return out

    def select_on_dev(self) -> dict:
        """Run the candidates once on the dev split and keep each level's best. The dev
        results go to the returned dict only, never to the archive."""
        self.slots = self.compute_slots()
        plan = self.dev_candidates()
        shadow = {r.key: ProgRecord(r.program, r.name, r.lineage, r.gen)
                  for cands in plan.values() for r in cands}
        for s in shadow.values():
            self.replay(s, self.dev, 0)
        self.allow_dev = True
        try:
            self.run_jobs([(s, q, 0) for s in shadow.values() for q in self.dev], "dev split")
        finally:
            self.allow_dev = False
        missing = {s.name: len(s.gaps(self.dev, 0)) for s in shadow.values() if s.gaps(self.dev, 0)}
        if missing:
            raise SystemExit(f"debates on the dev split could not be completed (server errors): {missing}; "
                             f"run again to finish them")
        result: dict = {"version": 4, "settings": self.settings, "generations": self.gen, "train": self.qids,
                        "dev": self.dev, "levels": {}}
        for name, cands in plan.items():
            rows = []
            for r in cands:
                s = shadow[r.key]
                rows.append({"name": r.name, "key": r.key, "program": r.program, "lineage": r.lineage,
                             "source": self.source_of(r), "gen": r.gen, "seed": r.gen == 0,
                             "train_acc": r.score(self.qids), "train_turns": r.turns(self.qids),
                             "train_tokens": r.tokens_used(self.qids), "dev_acc": s.score(self.dev),
                             "dev_turns": s.turns(self.dev, 0), "dev_tokens": s.tokens_used(self.dev),
                             "dev_marks": {q: s.reps[0][q][0] for q in self.dev}})
            chosen = min(rows, key=lambda d: (-round(d["dev_acc"], 9), d["dev_turns"] or 0.0,
                                              -round(d["train_acc"], 9), d["gen"], d["name"])) if rows else None
            result["levels"][name] = {"candidates": rows, "chosen": chosen["key"] if chosen else None,
                                      "chosen_name": chosen["name"] if chosen else None}
            log(f"{name}: " + ("no program" if not rows else "; ".join(
                f"{d['name']}{'*' if d['seed'] else ''} train {d['train_acc']:.3f} dev {d['dev_acc']:.3f}"
                f"@{d['dev_turns']:.1f}" for d in rows) + f" -> {chosen['name']}"))
        return result


def final_markdown(result: dict) -> str:
    lines = ["# Global pipeline: final programs (chosen on the dev split)", "",
             f"{len(result['dev'])} dev questions, {len(result['train'])} train questions, after generation "
             f"{result['generations']}. A * marks a seed. Train and dev scores hold selection luck: report the "
             f"test scores (eval_pipeline_global.py).", "",
             "| level | program | from seed | train acc | train turns | dev acc | dev turns | chosen |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for name, d in result["levels"].items():
        if not d["candidates"]:
            lines.append(f"| {name} | (no program reached this level) | | | | | | |")
        for c in d["candidates"]:
            lines.append(f"| {name} | {c['name']}{'*' if c['seed'] else ''} | {c['lineage']} ({c['source']}) "
                         f"| {c['train_acc']:.3f} "
                         f"| {c['train_turns']:.1f} | {c['dev_acc']:.3f} | {c['dev_turns']:.1f} | "
                         f"{'yes' if c['key'] == d['chosen'] else ''} |")
    lines += ["", "## Chosen programs", ""]
    for name, d in result["levels"].items():
        c = next((c for c in d["candidates"] if c["key"] == d["chosen"]), None)
        if c is not None:
            lines += [f"### {name}: {c['name']}", "```json", json.dumps(c["program"], indent=1), "```", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="run directory (archive, logs, results)")
    ap.add_argument("--splits", type=Path, required=True, help="the train / dev split (split_train_dev.py)")
    ap.add_argument("--seeds", type=Path, default=None, help="default: <out>/seeds.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_600_train.json",
                    help="holds every train and dev question")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--select-dev", action="store_true",
                    help="no search: run the final candidates on the dev split -> <out>/final.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--max-calls-per-question", type=int, default=None,
                    help="default: the turn cap (--turn-cap, 16 unless set)")
    ap.add_argument("--print-every", type=int, default=1)
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
    splits = load_split(args.splits)
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    lacking = [q for q in splits["train"] + splits["dev"] if q not in rows]
    if lacking:
        raise SystemExit(f"{len(lacking)} questions of {args.splits} are not in {args.dataset}")
    rows = {q: rows[q] for q in splits["train"] + splits["dev"]}
    P.check_rows(rows.values())             # the dataset suits the answer mode
    if args.seeds is None:
        args.seeds = Path(args.out) / "seeds.json"
    if args.live_cache is None:
        args.live_cache = Path(args.out).parent / f"rounds_{P.model_tag(args.model)}.jsonl"
    seeds = json.loads(args.seeds.read_text()) if args.seeds.exists() else None
    if seeds is None and not (args.resume or args.select_dev):
        raise SystemExit(f"{args.seeds} does not exist; run program_seeds_global.py first")
    if seeds is not None and seeds.get("settings") != settings:
        raise SystemExit(f"{args.seeds} was written under {seeds.get('settings')}, this run uses {settings}")
    if seeds is not None and set(seeds.get("examples") or []) - set(splits["train"]):
        raise SystemExit(f"{args.seeds} was written from example questions that are not in the train split "
                         f"of {args.splits}")
    seed_source = {s["name"]: s.get("source", "seed") for s in seeds["seeds"]} if seeds else {}
    Path(args.live_cache).parent.mkdir(parents=True, exist_ok=True)
    runner = P.make_runner(rows, args.live_cache, args.base_urls, args.model, args.temperature,
                           max_total_calls=args.max_total_calls, api_key=args.api_key,
                           lock=not args.ignore_cache_lock)
    log(f"executor {settings}; model {args.model}; cache {args.live_cache}")
    log(f"{len(splits['train'])} train questions (the search), {len(splits['dev'])} dev questions (the final "
        f"choice only); levels " + ", ".join(f"{n} {d}" for n, d in level_text(P.MAX_TURNS))
        + f"; {CHILDREN_PER_LEVEL} children per level and generation, {args.generations} generations")
    search = Search(args, rows, splits, runner, settings, seed_source)
    opened, stopped = False, False
    try:
        if args.select_dev:
            search.open_archive(resume=True)
            result = search.select_on_dev()
            (Path(args.out) / "final.json").write_text(json.dumps(result, indent=1))
            (Path(args.out) / "final.md").write_text(final_markdown(result))
            log(f"final programs -> {Path(args.out) / 'final.json'}, final.md")
            return
        search.open_archive(resume=args.resume)
        opened = True
        if args.resume:
            if seeds is not None:
                search.ensure_seeds(seeds["seeds"])
            search.settle_slots()
        else:
            log(f"generation 0: {len(seeds['seeds'])} seeds on {len(search.qids)} train questions")
            search.seed(seeds["seeds"])
        search.print_slots()
        while search.gen < args.generations:
            search.generation()
    except SF.BudgetExhausted as exc:
        log(f"\nstopped: {exc}")
        stopped = True
    except KeyboardInterrupt:
        log("\ninterrupted" + ("" if args.select_dev else "; the archive on disk is complete up to the last "
                                                          "finished wave"))
        stopped = True
    finally:
        try:
            if opened:
                search.slots = search.compute_slots()
                (Path(args.out) / "summary.json").write_text(json.dumps(search.summary(), indent=1))
                log(f"summary -> {Path(args.out) / 'summary.json'}")
                search.print_slots()
        finally:
            runner.close()
    if stopped and args.select_dev:
        raise SystemExit("the final choice on the dev split was not finished; final.json was not written")


if __name__ == "__main__":
    main()

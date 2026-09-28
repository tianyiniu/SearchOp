"""Per-group evolutionary search for debate-control programs.

One search over the six question groups of outputs/clusters_train_both.json
(50 search questions per group by default), keeping the best program per
group and cost band instead of one global winner. Nothing from earlier
experiments is used: no old cache, no old program. The executor runs the v2
pipeline (careful-reasoning prompts, 6144-token replies, a summary follow-up
that later speakers read) with a 300+900 digest window.

Scoring. A program is replayed on every search question; rounds are looked
up in this run's cache by (question, round sequence, replicate) and the
model is called only on a miss. Every result is stored per question id, so
the archive can be resumed and widened to more questions per group without
re-scoring what is already known.

Replicates. Replicate 0 screens every program. Replicate 1 is spent only to
confirm: programs that would be a cell elite or in a group's top five are
rerun on that group, and programs that solve a question few others solve are
rerun on those questions. A question's mark is the mean over its replicates.

Selection. Cells are group x cost band (cheap / medium / expensive by mean
speaker turns); each cell keeps its best program. Each generation makes 18
children: 6 from cell elites in rotation, 5 from lexicase selection over
every program ever scored, 5 from a farthest-point pass under a combined
structural + behavioural distance (within a quality floor), and 2 random
immigrants. At most a quarter of a generation's parents may share a seed
lineage. A child that behaves exactly like a program already in the archive
is a neutral edit and is mutated again.

Mutation. Random edits (rule operators plus a plan operator on the opening
rounds), and one third model-guided edits: the search draws the KIND of edit
and the guide model fills in the details from digests of the parent's
failures in a target group. Guided edits are checked against the drawn kind.

    python scripts/evolve_program_clusters.py \\
        --seeds outputs/cluster_search/seeds.json --out outputs/cluster_search/run1

    # later: continue, or widen to 100 per group, reusing every recording
    python scripts/evolve_program_clusters.py --out outputs/cluster_search/run1 --resume
    python scripts/evolve_program_clusters.py --out outputs/cluster_search/run1 --resume --per-group 100

    # champion picking on held-out questions (two replicates), after a search
    python scripts/evolve_program_clusters.py --out outputs/cluster_search/run1 --pick-champions
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_guide as G  # noqa: E402
import program_space as P  # noqa: E402
import schema_fitness as SF  # noqa: E402
from program_space import ProgRecord  # noqa: E402

ROOT = P.ROOT
BAND_NAMES = ("cheap", "medium", "expensive")


def log(msg: str = "") -> None:
    SF.log(msg)


class Search:
    def __init__(self, args, rows: dict, groups: dict, runner, guide_client, rng: random.Random,
                 settings: dict):
        self.args, self.rows, self.groups, self.runner = args, rows, groups, runner
        self.guide, self.rng, self.settings = guide_client, rng, settings
        self.gq: dict[int, list[str]] = {g["group"]: list(g["search"]) for g in groups["groups"]}
        self.group_ids = [g["group"] for g in groups["groups"]]
        self.qids: list[str] = [q for g in self.group_ids for q in self.gq[g]]
        self.group_of = {q: g for g in self.group_ids for q in self.gq[g]}
        self.group_text = {g["group"]: P.group_profile_text(g, P.difficulty_mix(rows, g["search"]))
                           for g in groups["groups"]}
        self.bands = [float(x) for x in args.cost_bands.split(",")]
        assert len(self.bands) == 2 and self.bands[0] < self.bands[1], args.cost_bands
        self.archive: dict[str, ProgRecord] = {}
        self.by_behaviour: dict[str, str] = {}
        self.cells: dict[tuple[int, int], str] = {}
        self.cell_order = [(g, b) for g in self.group_ids for b in range(3)]
        self.cell_ptr = 0
        self.gen = 0
        self.best_cell_score: dict[tuple[int, int], float] = {}
        self.last_improvement = 0
        self.guide_usage = {"calls": 0, "ok": 0, "fallback": Counter()}
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        self.archive_path = out / "archive.jsonl"
        self.log_path = out / "generations.jsonl"
        self._archive_fh = None

    # ------------------------------------------------------------------ archive
    def header(self) -> dict:
        return {"header": True, "settings": self.settings, "clusters": self.groups["source"],
                "per_group": self.groups["per_group"], "k": self.groups["k"],
                "qids": self.qids, "groups": {str(g): self.gq[g] for g in self.group_ids},
                "cost_bands": self.bands, "seed": self.args.seed}

    def open_archive(self, resume: bool) -> None:
        if resume:
            self.load_archive()
        else:
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
        for field in ("settings", "clusters", "k"):
            if header.get(field) != self.header()[field]:
                raise SystemExit(f"--resume: archive {field} {header.get(field)!r} differs from "
                                 f"this run's {self.header()[field]!r}")
        if header["per_group"] > self.groups["per_group"]:
            raise SystemExit("--resume: the archive was built with more questions per group "
                             "than this run asks for; --per-group can only grow")
        for d in recs.values():
            rec = ProgRecord.from_json(d)
            self.archive[rec.key] = rec
            self.gen = max(self.gen, rec.gen)
        self._archive_fh = self.archive_path.open("a")
        if header["per_group"] != self.groups["per_group"]:
            self._archive_fh.write(json.dumps(self.header()) + "\n")   # widened: new header
            self._archive_fh.flush()
        for rec in self.archive.values():               # file order = generation order
            rec.dup_of = None
            self.mark_duplicate(rec)
        log(f"resumed {len(self.archive)} programs from {self.archive_path} "
            f"(last generation {self.gen}; archive had {header['per_group']} per group, "
            f"this run {self.groups['per_group']})")

    def save(self, rec: ProgRecord) -> None:
        self._archive_fh.write(json.dumps(rec.to_json()) + "\n")
        self._archive_fh.flush()

    def add(self, rec: ProgRecord) -> None:
        self.archive[rec.key] = rec
        self.mark_duplicate(rec)
        self.save(rec)

    def mark_duplicate(self, rec: ProgRecord) -> None:
        """Once fully covered, a program whose behaviour matches an earlier
        program's is a duplicate individual: kept on file, never selected."""
        h = rec.behaviour(self.qids)
        if h is None:
            return
        owner = self.by_behaviour.setdefault(h, rec.key)
        rec.dup_of = None if owner == rec.key else owner

    def live(self) -> list[ProgRecord]:
        """The programs that may be selected: every archived program that is
        not a behavioural duplicate of an earlier one."""
        return [r for r in self.archive.values() if r.dup_of is None]

    # ------------------------------------------------------------------ scoring
    def replay(self, rec: ProgRecord, qids: list[str], rep: int) -> None:
        """Cache-only pass: nothing is spent."""
        self.runner.reset_budget(0)
        outs = P.run_many(rec.program, self.runner, self.rows, rec.gaps(qids, rep), rep,
                          self.args.max_calls_per_question, workers=1)
        for q, o in outs.items():
            if o is not None:
                rec.record(rep, q, o)

    def interleaved(self, qids: list[str]) -> list[str]:
        """`qids` reordered round-robin over groups, so a partial fill has
        questions from every group."""
        pos = {q: i for g in self.group_ids for i, q in enumerate(self.gq[g])}
        return sorted(qids, key=lambda q: (pos.get(q, 0), self.group_of.get(q, 0)))

    def fill(self, rec: ProgRecord, qids: list[str], rep: int, budget: int | None,
             est_turns: float, label: str = "") -> int:
        """Run `rec` on its uncovered questions among `qids` at `rep`. With a
        budget (speaker turns), admit as many questions as the budget affords
        at `est_turns` each and let every admitted question finish. Returns
        the turns spent."""
        gaps = self.interleaved(rec.gaps(qids, rep))
        if not gaps:
            return 0
        if budget is not None:
            n_admit = min(len(gaps), max(1, int(budget // max(est_turns, 1.0))))
            gaps = gaps[:n_admit]
            self.runner.reset_budget(budget + self.args.max_calls_per_question)
        else:
            self.runner.reset_budget(None)
        self.runner.stage(label or f"gen {self.gen} fill {rec.name}")
        outs = P.run_many(rec.program, self.runner, self.rows, gaps, rep,
                          self.args.max_calls_per_question, workers=self.args.workers)
        for q, o in outs.items():
            if o is not None:
                rec.record(rep, q, o)
        spent = self.runner.novel_calls
        self.runner.reset_budget(0)
        rec.live_turns += spent
        return spent

    def fill_pairs(self, pairs: list[tuple[ProgRecord, str]], rep: int, label: str) -> int:
        """Run many (program, question) pairs at `rep` in parallel, no budget.
        Used where the work is one question per program (rare-solve
        confirmation), which would otherwise run one debate at a time."""
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        pairs = [(r, q) for r, q in pairs if not r.has(rep, q)]
        if not pairs:
            return 0
        self.runner.reset_budget(None)
        self.runner.stage(label)
        cap = self.args.max_calls_per_question

        def one(pair):
            rec, q = pair
            try:
                return rec, q, M.run_program(rec.program, self.runner, self.rows[q], rep=rep, max_calls=cap)
            except M.OffCache:
                return rec, q, None

        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            results = list(tqdm(pool.map(one, pairs), total=len(pairs), unit="q", desc=label,
                                leave=False, disable=len(pairs) < 40))
        touched = {}
        for rec, q, o in results:
            if o is not None:
                rec.record(rep, q, o)
            touched[rec.key] = rec
        spent = self.runner.novel_calls
        self.runner.reset_budget(0)
        for rec in touched.values():
            self.add(rec)
        return spent

    def band(self, turns: float | None) -> int:
        if turns is None:
            return 1
        return 0 if turns < self.bands[0] else 1 if turns <= self.bands[1] else 2

    def cell_score(self, g: int, b: int) -> float | None:
        key = self.cells.get((g, b))
        return self.archive[key].score(self.gq[g]) if key else None

    def passes_screen(self, rec: ProgRecord, est_turns: float) -> tuple[bool, str]:
        """Whether an uncovered program is worth spending on: a first look is
        always allowed; after that, only if its projected score on some group,
        after the fills the budget affords, is within the margin of that
        group's elite in its cost band."""
        gaps = rec.gaps(self.qids, 0)
        covered = len(self.qids) - len(gaps)
        if covered == 0:
            return True, "first look"
        turns = rec.turns(self.qids) or est_turns
        n_fill = min(len(gaps), max(1, int(self.args.novelty_budget // max(turns, 1.0))))
        share = n_fill / len(gaps)
        for g in self.group_ids:
            gq = self.gq[g]
            g_gaps = [q for q in gq if not rec.has(0, q)]
            g_cov = len(gq) - len(g_gaps)
            if g_cov == 0:
                continue
            acc = rec.n_correct([q for q in gq if rec.has(0, q)]) / g_cov
            projected = (acc * g_cov + acc * share * len(g_gaps)) / len(gq)
            elite = self.cell_score(g, self.band(rec.turns(gq) or turns))
            if elite is None or projected >= elite - self.args.prescreen_margin:
                return True, f"group {g} projected {projected:.2f} vs elite {elite}"
        return False, "no group within margin"

    def score_new(self, rec: ProgRecord, est_turns: float) -> str:
        """Replicate-0 score of a new program: free replay, then a screened fill."""
        self.replay(rec, self.qids, 0)
        if not rec.gaps(self.qids, 0):
            return "cached"
        ok, why = self.passes_screen(rec, est_turns)
        if not ok:
            return "partial:" + why
        spent = self.fill(rec, self.qids, 0, self.args.novelty_budget, est_turns)
        return ("full" if not rec.gaps(self.qids, 0) else "partial") + f":{spent} turns"

    def top_up(self, all_ages: bool = False) -> int:
        """Give recent partial programs (or, after widening, every partial
        program) another screened fill."""
        spent = 0
        for rec in list(self.archive.values()):
            if not rec.gaps(self.qids, 0):
                continue
            if not all_ages and rec.gen < self.gen - self.args.topup_window:
                continue
            self.replay(rec, self.qids, 0)                    # others may have recorded its paths
            if not rec.gaps(self.qids, 0):
                self.add(rec)
                continue
            ok, _ = self.passes_screen(rec, rec.turns(self.qids) or 6.0)
            if ok:
                spent += self.fill(rec, self.qids, 0, self.args.novelty_budget,
                                   rec.turns(self.qids) or 6.0, label=f"gen {self.gen} top-up")
                self.add(rec)
        return spent

    # --------------------------------------------------------------- cells etc.
    def full_on(self, rec: ProgRecord, g: int) -> bool:
        return rec.covered(self.gq[g]) == len(self.gq[g])

    def recompute_cells(self, track: bool = True) -> bool:
        """Elite per (group, band). Returns True if any cell's best score so far
        improved; with track=False the cells are refreshed (e.g. so the
        confirmation step knows the provisional elites) without recording that."""
        new: dict[tuple[int, int], tuple[float, float, str]] = {}
        for rec in self.live():
            for g in self.group_ids:
                if not self.full_on(rec, g):
                    continue
                t = rec.turns(self.gq[g])
                b = self.band(t)
                cand = (rec.score(self.gq[g]), -(t or 0.0), rec.key)
                if (g, b) not in new or cand[:2] > new[(g, b)][:2]:
                    new[(g, b)] = cand
        improved = False
        self.cells = {}
        for cell, (s, _, key) in new.items():
            self.cells[cell] = key
            if track and s > self.best_cell_score.get(cell, -1.0) + 1e-9:
                self.best_cell_score[cell] = s
                improved = True
        return improved

    def confirm(self) -> int:
        """Replicate 1 where it decides something: candidate elites and top-5
        per group on that group's questions; rare solves on those questions."""
        spent = 0
        for g in self.group_ids:
            gq = self.gq[g]
            ranked = sorted((r for r in self.live() if self.full_on(r, g)),
                            key=lambda r: (-r.score(gq), r.turns(gq) or 0.0))
            cands = {r.key for r in ranked[: self.args.top_n]}
            cands |= {self.cells[c] for c in self.cells if c[0] == g}
            for key in cands:
                rec = self.archive[key]
                if rec.gaps(gq, 1):
                    spent += self.fill(rec, gq, 1, None, rec.turns(gq) or 6.0,
                                       label=f"gen {self.gen} confirm g{g} {rec.name}")
                    self.add(rec)
        # rare solves: a question solved (at replicate 0) by at most rare_k programs
        solvers: dict[str, list[str]] = defaultdict(list)
        for rec in self.live():
            r0 = rec.reps.get(0, {})
            for q in self.qids:
                if q in r0 and r0[q][0] == 1:
                    solvers[q].append(rec.key)
        pairs = [(self.archive[key], q) for q, keys in solvers.items()
                 if len(keys) <= self.args.rare_k for key in keys]
        spent += self.fill_pairs(pairs, 1, f"gen {self.gen} confirm rare solves")
        return spent

    # ------------------------------------------------------------- selection
    def lexicase(self, pool: list[ProgRecord]) -> ProgRecord | None:
        if not pool:
            return None
        cases = list(self.qids)
        self.rng.shuffle(cases)
        alive = list(pool)
        for q in cases:
            best = max((r.mark(q) or 0.0) for r in alive)
            alive = [r for r in alive if (r.mark(q) or 0.0) >= best - 1e-9]
            if len(alive) == 1:
                break
        return self.rng.choice(alive)

    def weakest_group(self, rec: ProgRecord) -> int:
        """The group with the most room between this program and the group's
        best elite; the target of a guided edit."""
        room = []
        for g in self.group_ids:
            best = max((self.cell_score(g, b) or 0.0) for b in range(3))
            room.append((best - rec.score(self.gq[g]), self.rng.random(), g))
        return max(room)[2]

    def pick_parents(self) -> list[tuple[ProgRecord, str, int]]:
        """(parent, slot kind, target group) for each mutated child."""
        n_slots = self.args.children - self.args.immigrants
        cap = max(1, math.ceil(n_slots / 4))
        chosen: list[tuple[ProgRecord, str, int]] = []
        lineage = Counter()

        def take(rec: ProgRecord, kind: str, g: int | None) -> None:
            chosen.append((rec, kind, g if g is not None else self.weakest_group(rec)))
            lineage[rec.lineage] += 1

        def allowed(rec: ProgRecord) -> bool:
            return lineage[rec.lineage] < cap

        occupied = [c for c in self.cell_order if c in self.cells]
        n_elite = min(self.args.elite_slots, len(occupied), n_slots)
        for _ in range(n_elite):
            for _ in range(len(occupied)):
                c = occupied[self.cell_ptr % len(occupied)]
                self.cell_ptr += 1
                rec = self.archive[self.cells[c]]
                # one program may hold several cells; it parents once per generation
                if allowed(rec) and rec.key not in {r.key for r, _, _ in chosen}:
                    take(rec, "elite", c[0])
                    break
        n_lex = min(self.args.lexicase_slots, n_slots - len(chosen))
        for _ in range(n_lex):
            rec = self.lexicase([r for r in self.live() if allowed(r)])
            if rec is not None:
                take(rec, "lexicase", None)
        # diversity picks: within the quality floor of some cell elite, farthest
        # (combined distance) from the parents already chosen
        floor_ok = []
        already = {r.key for r, _, _ in chosen}
        for rec in self.live():
            if not allowed(rec) or rec.key in already:
                continue
            for g in self.group_ids:
                if not self.full_on(rec, g):
                    continue
                elite = self.cell_score(g, self.band(rec.turns(self.gq[g])))
                if elite is None or rec.score(self.gq[g]) >= elite - self.args.floor:
                    floor_ok.append(rec)
                    break
        n_div = min(self.args.diversity_slots, n_slots - len(chosen))
        if n_div > 0 and floor_ok:
            fixed = [r for r, _, _ in chosen]
            picked = P.farthest_point(floor_ok, n_div,
                                      lambda a, b: P.combined_distance(a, b, self.qids), fixed=fixed)
            for i in picked:
                if allowed(floor_ok[i]):
                    take(floor_ok[i], "diversity", None)
        while len(chosen) < n_slots:                       # leftovers go to lexicase
            rec = self.lexicase([r for r in self.live() if allowed(r)])
            if rec is None:
                rec = self.lexicase(self.live())
                if rec is None:
                    break
            take(rec, "lexicase", None)
        return chosen

    # -------------------------------------------------------------- children
    def make_child(self, parent: ProgRecord, target_g: int, guided: bool, idx: int,
                   stats: Counter) -> ProgRecord | None:
        """One child of `parent`, scored at replicate 0. Neutral children (same
        behaviour as something already scored) are mutated again, randomly."""
        seen = set(self.archive)
        est = parent.turns(self.qids) or 6.0
        for attempt in range(self.args.neutral_retries + 1):
            op = P.draw_op(self.rng, self.args.plan_prob)
            child = None
            was_guided = False
            if guided and attempt == 0 and self.guide is not None:
                stats["guided_tried"] += 1
                self.guide_usage["calls"] += 1
                scores = [f"  group {g}: {parent.score(self.gq[g]):.2f}, "
                          f"{(parent.turns(self.gq[g]) or 0):.1f} turns" for g in self.group_ids]
                digests = G.build_digests(parent.program, parent, self.runner, self.rows,
                                          self.gq[target_g], target_g, self.rng,
                                          self.args.n_fail_digests, self.args.n_ok_digests,
                                          max_calls=self.args.max_calls_per_question)
                child, why = G.guided_edit(self.guide, parent.program, op, scores,
                                           self.group_text[target_g], digests,
                                           effort=self.args.guide_effort)
                if child is None:
                    stats["guided_fallback"] += 1
                    self.guide_usage["fallback"][why.split(":")[0][:40]] += 1
                else:
                    was_guided = True
                    self.guide_usage["ok"] += 1
                    op_name = op
            if child is None:
                child, op_name = P.mutate(parent.program, self.rng, op)
                if op_name == "none":
                    stats["mutation_failed"] += 1
                    continue
            key = P.canon(child)
            if key in seen:
                stats["duplicate_text"] += 1
                continue
            seen.add(key)
            rec = ProgRecord(child, f"g{self.gen}_{idx}" + ("_r" * attempt), parent.lineage,
                             self.gen, parent=parent.key, op=op_name, guided=was_guided)
            status = self.score_new(rec, est)
            h = rec.behaviour(self.qids)
            if h is not None and h in self.by_behaviour:
                stats["neutral"] += 1
                continue
            stats["op:" + P.op_family(op_name)] += 1
            stats["guided_child" if was_guided else "random_child"] += 1
            stats["status:" + status.split(":")[0]] += 1
            return rec
        return None

    def immigrant(self, idx: int, stats: Counter) -> ProgRecord | None:
        pool = []
        seen = set(self.archive)
        while len(pool) < self.args.immigrant_pool:
            prog = P.random_program(self.rng)
            if P.canon(prog) not in seen:
                seen.add(P.canon(prog))
                pool.append(prog)
        toks = [P.struct_tokens(p) for p in pool]
        fixed = [r.tokens for r in self.archive.values()]
        order = P.farthest_point(toks, min(4, len(toks)), P.struct_distance, fixed=fixed)
        for attempt, i in enumerate(order):          # a duplicate individual is not stored
            rec = ProgRecord(pool[i], f"g{self.gen}_imm{idx}" + ("_r" * attempt),
                             f"immigrant_g{self.gen}_{idx}", self.gen, op="immigrant")
            self.score_new(rec, 6.0)
            h = rec.behaviour(self.qids)
            if h is None or h not in self.by_behaviour:
                return rec
            stats["neutral_immigrant"] += 1
        return None

    # ------------------------------------------------------------ the loop
    def seed(self, seeds: list[dict]) -> None:
        """Generation 0: every seed on every search question, no budget."""
        self.gen = 0
        for s in seeds:
            prog = P.normalize_program(s["program"])
            P.validate_program(prog)
            key = P.canon(prog)
            if key in self.archive:
                rec = self.archive[key]
            else:
                rec = ProgRecord(prog, s["name"], s["name"], 0, op=s.get("source", "seed"))
            self.replay(rec, self.qids, 0)
            self.fill(rec, self.qids, 0, None, 6.0, label=f"seed {s['name']}")
            self.add(rec)
            log(f"  seed {s['name']:24s} acc {rec.score(self.qids):.3f}  "
                f"turns {rec.turns(self.qids) or 0:.1f}  live turns {rec.live_turns}")
        self.recompute_cells(track=False)       # provisional elites, so confirm() sees them
        spent = self.confirm()
        self.recompute_cells()
        self.last_improvement = self.gen
        self.write_gen_log({"gen": 0, "seeds": len(seeds), "confirm_turns": spent})
        self.print_cells()

    def ensure_seeds(self, seeds: list[dict]) -> int:
        """On resume: add any seed the archive lacks (an interrupted seed stage
        leaves its recordings in the cache, so this is mostly free)."""
        missing = [s for s in seeds if P.canon(P.normalize_program(s["program"])) not in self.archive]
        for s in missing:
            prog = P.normalize_program(s["program"])
            P.validate_program(prog)
            rec = ProgRecord(prog, s["name"], s["name"], 0, op=s.get("source", "seed"))
            self.replay(rec, self.qids, 0)
            self.fill(rec, self.qids, 0, None, 6.0, label=f"seed {s['name']}")
            self.add(rec)
        return len(missing)

    def widen(self) -> None:
        """After a resume with more questions per group: seeds and current
        elites are completed on the new questions; everything else is a
        partial program that the top-up screens."""
        keys = {k for k in self.archive if self.archive[k].gen == 0}
        self.recompute_cells()
        keys |= set(self.cells.values())
        for key in keys:
            rec = self.archive[key]
            self.replay(rec, self.qids, 0)
            if rec.gaps(self.qids, 0):
                self.fill(rec, self.qids, 0, None, rec.turns(self.qids) or 6.0,
                          label=f"widen {rec.name}")
                self.add(rec)
        self.recompute_cells(track=False)       # elites on the new questions, for the screen
        self.top_up(all_ages=True)
        self.recompute_cells(track=False)
        self.confirm()
        self.recompute_cells()
        self.last_improvement = self.gen

    def generation(self) -> None:
        self.gen += 1
        t0 = time.perf_counter()
        stats: Counter = Counter()
        before = dict(self.cells)
        stats["topup_turns"] = self.top_up()
        parents = self.pick_parents()
        n_mut = len(parents)
        n_guided = round(n_mut * self.args.guided_frac) if self.guide is not None else 0
        flags = [True] * n_guided + [False] * (n_mut - n_guided)
        self.rng.shuffle(flags)
        parent_lineages = Counter(r.lineage for r, _, _ in parents)
        slot_kinds = Counter(k for _, k, _ in parents)
        children: list[ProgRecord] = []
        for i, ((parent, kind, g), guided) in enumerate(zip(parents, flags)):
            self.runner.stage(f"gen {self.gen} child {i + 1}/{n_mut}")
            rec = self.make_child(parent, g, guided, i, stats)
            if rec is not None:
                stats["rules_delta_" + ("guided" if rec.guided else "random")] += (
                    len(rec.program["rules"]) - len(parent.program["rules"]))
                self.add(rec)
                children.append(rec)
        for j in range(self.args.immigrants):
            rec = self.immigrant(j, stats)
            if rec is not None:
                self.add(rec)
                children.append(rec)
                stats["immigrants"] += 1
        self.recompute_cells(track=False)       # provisional elites, so confirm() sees them
        stats["confirm_turns"] = self.confirm()
        improved = self.recompute_cells()
        if improved:
            self.last_improvement = self.gen
        new_elites = [k for c, k in self.cells.items() if before.get(c) != k]
        for k in new_elites:
            r = self.archive[k]
            stats["new_elite_" + ("guided" if r.guided else "immigrant" if r.op == "immigrant"
                                  else "random")] += 1
        entry = {"gen": self.gen, "children": len(children), "parents": slot_kinds,
                 "lineages": parent_lineages, "cells_occupied": len(self.cells),
                 "new_elites": len(new_elites), "improved": improved,
                 "archive": len(self.archive), "live_calls": getattr(self.runner, "calls", 0),
                 "summaries": getattr(self.runner, "summaries", 0),
                 "errors": getattr(self.runner, "errors", 0),
                 "v2": dict(D.V2_STATS),
                 "guide": {"calls": self.guide_usage["calls"], "ok": self.guide_usage["ok"]},
                 "seconds": round(time.perf_counter() - t0, 1), **stats}
        self.write_gen_log(entry)
        log(f"gen {self.gen:>3}: {len(children)} children "
            f"(guided {stats['guided_child']}, neutral {stats['neutral']}, "
            f"dup {stats['duplicate_text']}), cells {len(self.cells)}/{len(self.cell_order)}, "
            f"new elites {len(new_elites)}, archive {len(self.archive)}, "
            f"live calls {entry['live_calls']}, {entry['seconds']}s")
        if self.gen % self.args.print_every == 0:
            self.print_cells()

    def write_gen_log(self, entry: dict) -> None:
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(entry, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
                     + "\n")

    def print_cells(self) -> None:
        log(f"  {'group':>5} {'band':>9} {'acc':>6} {'turns':>6} {'reps':>4}  program")
        for (g, b) in self.cell_order:
            key = self.cells.get((g, b))
            if key is None:
                log(f"  {g:>5} {BAND_NAMES[b]:>9} {'-':>6}")
                continue
            r = self.archive[key]
            gq = self.gq[g]
            reps = sum(1 for rep in r.reps if all(r.has(rep, q) for q in gq))
            log(f"  {g:>5} {BAND_NAMES[b]:>9} {r.score(gq):>6.3f} {r.turns(gq) or 0:>6.1f} {reps:>4}  "
                f"{r.name} ({r.lineage})")

    def summary(self) -> dict:
        cells = {}
        for (g, b), key in self.cells.items():
            r = self.archive[key]
            cells[f"{g}:{BAND_NAMES[b]}"] = {"name": r.name, "lineage": r.lineage, "key": key,
                                             "score": r.score(self.gq[g]), "turns": r.turns(self.gq[g]),
                                             "program": r.program}
        overall = sorted(((r.score(self.qids), r.key) for r in self.live()
                          if not r.gaps(self.qids, 0)), reverse=True)[:5]
        dups = sum(1 for r in self.archive.values() if r.dup_of is not None)
        return {"generations": self.gen, "archive": len(self.archive), "duplicates": dups, "cells": cells,
                "top_overall": [{"score": s, "name": self.archive[k].name,
                                 "program": self.archive[k].program} for s, k in overall],
                "live_calls": getattr(self.runner, "calls", 0), "guide": {
                    "calls": self.guide_usage["calls"], "ok": self.guide_usage["ok"],
                    "fallback": dict(self.guide_usage["fallback"])}}

    # ------------------------------------------------------------ champions
    def pick_champions(self) -> dict:
        """Per group: finalists = cell elites plus the next best by standing
        score, five in all, distinct in behaviour; each runs on the group's
        held-out questions at two replicates; the champion is the best held-out
        mean, ties to fewer turns. The same over all search questions picks a
        global champion on the union of held-out samples."""
        self.recompute_cells()
        hrng = random.Random(self.args.seed + 7)
        held: dict[int, list[str]] = {}
        for g in self.groups["groups"]:
            pool = [q for q in g["held_out"] if q in self.rows]
            n = min(len(pool), self.args.heldout_cap)
            held[g["group"]] = sorted(hrng.sample(pool, n))
        result: dict = {"per_group": {}, "heldout_cap": self.args.heldout_cap}

        def finalists(g: int | None) -> list[ProgRecord]:
            qids = self.qids if g is None else self.gq[g]
            picked: list[ProgRecord] = []
            seen_beh: set[str] = set()
            firsts = [] if g is None else [self.archive[self.cells[c]] for c in self.cell_order
                                           if c in self.cells and c[0] == g]
            ranked = sorted((r for r in self.live() if not r.gaps(qids, 0)),
                            key=lambda r: (-r.score(qids), r.turns(qids) or 0.0))
            for r in firsts + ranked:
                h = r.behaviour(qids)
                if r.key in {p.key for p in picked} or (h is not None and h in seen_beh):
                    continue
                picked.append(r)
                if h is not None:
                    seen_beh.add(h)
                if len(picked) == self.args.top_n:
                    break
            return picked

        def evaluate(rec: ProgRecord, qids: list[str]) -> dict:
            self.runner.reset_budget(None)
            per_rep = {}
            for rep in range(self.args.reps):
                self.runner.stage(f"champion {rec.name} rep {rep}")
                outs = P.run_many(rec.program, self.runner, self.rows, qids, rep,
                                  self.args.max_calls_per_question, workers=self.args.workers)
                done = {q: o for q, o in outs.items() if o is not None}
                per_rep[str(rep)] = {"acc": sum(o["correct"] for o in done.values()) / max(len(qids), 1),
                                     "turns": (sum(o["n_calls"] for o in done.values()) / len(done)) if done else None,
                                     "n": len(done),
                                     "marks": {q: int(o["correct"]) for q, o in done.items()}}
            self.runner.reset_budget(0)
            accs = [v["acc"] for v in per_rep.values()]
            return {"name": rec.name, "lineage": rec.lineage, "key": rec.key, "program": rec.program,
                    "held_out_acc": sum(accs) / len(accs), "per_rep": per_rep,
                    "turns": per_rep["0"]["turns"]}

        for g in self.group_ids:
            rows_g = [dict(evaluate(r, held[g]), search_score=r.score(self.gq[g]),
                           search_turns=r.turns(self.gq[g])) for r in finalists(g)]
            rows_g.sort(key=lambda d: (-d["held_out_acc"], d["turns"] or 0.0))
            result["per_group"][str(g)] = {"held_out_n": len(held[g]), "finalists": rows_g,
                                           "champion": rows_g[0]["key"] if rows_g else None}
            log(f"group {g}: held-out {len(held[g])} q; " + "; ".join(
                f"{d['name']} {d['held_out_acc']:.3f}@{(d['turns'] or 0):.1f}" for d in rows_g))
        all_held = sorted({q for qs in held.values() for q in qs})
        rows_all = [dict(evaluate(r, all_held), search_score=r.score(self.qids),
                         search_turns=r.turns(self.qids)) for r in finalists(None)]
        rows_all.sort(key=lambda d: (-d["held_out_acc"], d["turns"] or 0.0))
        result["global"] = {"held_out_n": len(all_held), "finalists": rows_all,
                            "champion": rows_all[0]["key"] if rows_all else None}
        log("global: " + "; ".join(f"{d['name']} {d['held_out_acc']:.3f}@{(d['turns'] or 0):.1f}"
                                   for d in rows_all))
        return result


def archive_per_group(path: Path) -> int | None:
    """The questions-per-group of an archive's latest header, or None."""
    if not path.exists():
        return None
    per_group = None
    for line in path.open():
        if line.startswith('{"header": true'):
            try:
                per_group = json.loads(line)["per_group"]
            except (ValueError, KeyError):
                pass
    return per_group


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="run directory (archive, logs, results)")
    ap.add_argument("--seeds", type=Path, default=ROOT / "outputs/cluster_search/seeds.json")
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs/clusters_train_both.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_program_search_train.json")
    ap.add_argument("--per-group", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pick-champions", action="store_true", help="no search: champion picking only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10, help="stop after this many generations without a cell improving")
    ap.add_argument("--children", type=int, default=18)
    ap.add_argument("--immigrants", type=int, default=2)
    ap.add_argument("--elite-slots", type=int, default=6)
    ap.add_argument("--lexicase-slots", type=int, default=5)
    ap.add_argument("--diversity-slots", type=int, default=5)
    ap.add_argument("--guided-frac", type=float, default=1 / 3)
    ap.add_argument("--plan-prob", type=float, default=0.2, help="share of edits that change the opening rounds")
    ap.add_argument("--cost-bands", default="4,8", help="mean turns: cheap < a <= medium <= b < expensive")
    ap.add_argument("--floor", type=float, default=0.08, help="quality floor below a cell elite for diversity picks")
    ap.add_argument("--top-n", type=int, default=5, help="per-group top-N that get replicate 1 / are finalists")
    ap.add_argument("--rare-k", type=int, default=3, help="a question solved by <= this many programs is rare")
    ap.add_argument("--neutral-retries", type=int, default=3)
    ap.add_argument("--immigrant-pool", type=int, default=30)
    ap.add_argument("--topup-window", type=int, default=5, help="partial programs younger than this get re-screened")
    ap.add_argument("--novelty-budget", type=int, default=300,
                    help="speaker turns one screened fill may spend (a first look at a program "
                         "with no cached path covers about this / mean-turns questions)")
    ap.add_argument("--prescreen-margin", type=float, default=0.02)
    ap.add_argument("--max-calls-per-question", type=int, default=16)
    ap.add_argument("--n-fail-digests", type=int, default=5)
    ap.add_argument("--n-ok-digests", type=int, default=2)
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--reps", type=int, default=2, help="replicates for champion picking")
    ap.add_argument("--heldout-cap", type=int, default=150, help="held-out questions per group used for picking")
    guide = ap.add_argument_group("guide model")
    guide.add_argument("--no-guide", action="store_true")
    guide.add_argument("--guide-effort", default=G.GUIDE_EFFORT)
    live = ap.add_argument_group("debate model")
    live.add_argument("--base-urls", default=P.DEFAULT_BASE_URLS)
    live.add_argument("--model", default=P.DEFAULT_MODEL)
    live.add_argument("--api-key", default="EMPTY")
    live.add_argument("--temperature", type=float, default=0.7)
    live.add_argument("--workers", type=int, default=64,
                      help="debates in flight at once; each may have up to 4 speakers in flight")
    live.add_argument("--live-cache", type=Path, default=None,
                      help="round cache (default outputs/cluster_search/rounds_<model>.jsonl, shared by seeds and runs)")
    live.add_argument("--max-total-calls", type=int, default=400_000)
    live.add_argument("--ignore-cache-lock", action="store_true")
    ap.add_argument("--digest-head", type=int, default=P.DIGEST_HEAD)
    ap.add_argument("--digest-tail", type=int, default=P.DIGEST_TAIL)
    ap.add_argument("--visible-reasoning", action="store_true",
                    help="for models that think in a hidden channel (gpt-oss): ask every speaker to "
                         "write its reasoning in the visible reply; changes prompts and cache keys")
    args = ap.parse_args()

    settings = P.configure_executor(args.digest_head, args.digest_tail, args.visible_reasoning)
    settings["model"] = args.model
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    if args.pick_champions:                       # the archive's question set, whatever it is
        args.per_group = archive_per_group(Path(args.out) / "archive.jsonl") or args.per_group
    groups = P.load_groups(args.clusters, args.per_group)
    if args.live_cache is None:
        args.live_cache = ROOT / "outputs/cluster_search" / f"rounds_{P.model_tag(args.model)}.jsonl"
    rng = random.Random(args.seed)
    guide_client = None if args.no_guide else G.make_client()
    runner = P.make_runner(rows, args.live_cache, args.base_urls, args.model, args.temperature,
                           max_total_calls=args.max_total_calls, api_key=args.api_key,
                           lock=not args.ignore_cache_lock)
    log(f"executor: v2, digest {settings['digest']}, {settings['answer_tokens']}-token replies, "
        f"model {args.model}; cache {args.live_cache}")
    log(f"{len(groups['groups'])} groups x {args.per_group} = {sum(len(g['search']) for g in groups['groups'])} "
        f"search questions; held out " + ", ".join(str(len(g['held_out'])) for g in groups['groups']))
    search = Search(args, rows, groups, runner, guide_client, rng, settings)
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
            if args.seeds.exists():
                n = search.ensure_seeds(json.loads(args.seeds.read_text())["seeds"])
                if n:
                    log(f"added {n} seeds the archive lacked")
            search.widen()
            search.print_cells()
        else:
            seeds = json.loads(args.seeds.read_text())
            if seeds.get("settings", {}).get("model") not in (None, args.model):
                log(f"note: seeds were sanity-run on {seeds['settings'].get('model')}, this run uses {args.model}")
            log(f"generation 0: {len(seeds['seeds'])} seeds on {len(search.qids)} questions")
            search.seed(seeds["seeds"])
        while search.gen < args.generations:
            if search.gen - search.last_improvement >= args.patience and search.gen > 0:
                log(f"no cell improved for {args.patience} generations; stopping")
                break
            search.generation()
    except SF.BudgetExhausted as exc:
        log(f"\nstopped: {exc}")
    except KeyboardInterrupt:
        log("\ninterrupted; the archive on disk is complete up to the last saved program")
    finally:
        try:
            if opened:
                search.recompute_cells()
                summary = search.summary()
                (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1))
                log(f"summary -> {Path(args.out) / 'summary.json'}")
                search.print_cells()
        finally:
            runner.close()


if __name__ == "__main__":
    main()

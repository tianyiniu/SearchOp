"""Offline checks for the v3 per-group search (sample_cluster_subsets,
program_seeds_v3, evolve_program_clusters_v3). Reuses the fake debate model
and fake guide model of test_program_clusters.py, so the v2 executor, the
round cache and its keys run exactly as they would live. No vLLM, no API key.

    python scripts/test_program_clusters_v3.py
"""

from __future__ import annotations

import json
import random
import sys
import types
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_program_clusters as T  # noqa: E402  (installs the fake clients on import)
import program_guide as G  # noqa: E402
import program_space as P  # noqa: E402
import evolve_program_clusters_v3 as V  # noqa: E402
import program_seeds_v3 as S3  # noqa: E402
import sample_cluster_subsets as SUB  # noqa: E402

check, run_cli, TMP = T.check, T.run_cli, T.TMP
CLUSTERS_V3 = TMP / "clusters_v3.json"
CACHE = TMP / "rounds_v3.jsonl"

# the fake guide learns the one-program-per-group request
_old_parse = T.FakeResponses.parse


def _parse(self, model, instructions, input, reasoning, text_format, max_output_tokens):
    if text_format is G.GroupSeedBatch:
        self.calls += 1
        todo = json.loads(input.split("for each of these groups: ", 1)[1].split(".", 1)[0])
        progs = []
        for g in todo:
            child, _ = P.mutate_uniform(P.PROTOCOLS["fresh_on_disagree"], self.rng)
            progs.append(G.GroupSeedProgram(group=g, name=f"fake_{g}", strategy="a fake variant",
                                            program=G.Program(**child)))
        return T.FakeParsed(G.GroupSeedBatch(programs=progs))
    return _old_parse(self, model, instructions, input, reasoning, text_format, max_output_tokens)


T.FakeResponses.parse = _parse


def test_statistics():
    print("statistics")
    mean, se = V.paired_se([1, 1, 0, 0], [1, 0, 0, 1])
    check(abs(mean) < 1e-12 and abs(se - (2 / 3) ** 0.5 / 2) < 1e-9, "paired standard error", f"{mean}, {se:.4f}")
    pct = V.percentile_ranks({"a": 0.1, "b": 0.3, "c": 0.3, "d": 0.5})
    check(pct["a"] < pct["b"] == pct["c"] < pct["d"] and abs(pct["b"] - 0.5) < 1e-9,
          "percentile ranks share ties", str(pct))


def test_edits():
    print("uniform edits")
    rng = random.Random(1)
    direct = P.normalize_program(P.PROTOCOLS["direct"])
    kinds = P.applicable_edits(direct)
    check("drop_rule" not in kinds and "swap_rules" not in kinds and "plan_drop" not in kinds
          and "plan_add" in kinds, "inapplicable kinds are left out", str(kinds))
    seen = Counter()
    prog = P.normalize_program(P.PROTOCOLS["early_exit_agree"])
    for _ in range(600):
        child, op = P.mutate_uniform(prog, rng)
        P.validate_program(child)
        seen[op] += 1
    n_kinds = len(P.applicable_edits(prog))
    check(set(seen) == set(P.applicable_edits(prog)), "every applicable kind is drawn", str(dict(seen)))
    check(min(seen.values()) > 0.5 * 600 / n_kinds, "kinds are drawn about equally", str(dict(seen)))
    old = random.Random(3)
    new = random.Random(3)
    same = all(P.mutate_plan(prog, old) == P.mutate_plan(prog, new, None) for _ in range(20))
    check(same, "the v2 plan edit is unchanged when no kind is named")


def test_subsets():
    print("search questions")
    run_cli(SUB, ["--out", str(CLUSTERS_V3), "--per-cluster", "50"])
    src = json.loads((P.ROOT / "outputs/clusters_train_both.json").read_text())
    d = json.loads(CLUSTERS_V3.read_text())
    check([c["members"] for c in d["clusters"]] == [c["members"] for c in src["clusters"]],
          "the groups themselves are unchanged")
    ok = all(len(c["subset"]) == 50 and len(set(c["subset"])) == 50
             and not set(c["subset"]) & set(c["held_out"])
             and set(c["subset"]) | set(c["held_out"]) == set(c["members"]) for c in d["clusters"])
    check(ok, "50 per group, disjoint from held-out, together the whole group")
    closer = [sum(c["subset_rank"].values()) / 50 < (c["size"] + 1) / 2 for c in d["clusters"]]
    check(sum(closer) >= 5, "the draw leans towards the centre", str(closer))
    first = [c["subset"] for c in d["clusters"]]
    run_cli(SUB, ["--out", str(CLUSTERS_V3), "--per-cluster", "50", "--force"])
    check(first == [c["subset"] for c in json.loads(CLUSTERS_V3.read_text())["clusters"]],
          "the draw is fixed by its seed")


def test_seeds() -> Path:
    print("seed stage")
    check(S3.seed_counts(6) == (6, 8, 10) and S3.seed_counts(2) == (2, 6, 0)
          and S3.seed_counts(10) == (10, 8, 22), "seed counts follow k",
          f"{S3.seed_counts(6)} {S3.seed_counts(2)} {S3.seed_counts(10)}")
    out = TMP / "v3" / "seeds.json"
    run_cli(S3, ["--clusters", str(CLUSTERS_V3), "--per-group", "3", "--out", str(out),
                 "--live-cache", str(CACHE), "--sanity-per-group", "1", "--min-turns", "1",
                 "--workers", "8"])
    seeds = json.loads(out.read_text())["seeds"]
    src = Counter(s["source"] for s in seeds)
    check(len(seeds) == 24 and src == {"protocol": 8, "llm": 6, "random": 10}, "24 seeds = 8 + 6 + 10",
          str(dict(src)))
    check(sorted(s["group"] for s in seeds if s["source"] == "llm") == list(range(6)),
          "one model-written seed per group")
    check(len({P.canon(s["program"]) for s in seeds}) == 24, "all seeds differ")
    return out


def test_search(seeds_path: Path):
    print("search")
    run = TMP / "v3" / "run"
    common = ["--seeds", str(seeds_path), "--out", str(run), "--clusters", str(CLUSTERS_V3),
              "--per-group", "3", "--live-cache", str(CACHE), "--workers", "8"]
    run_cli(V, common + ["--generations", "2"])
    lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    header, recs = lines[0], {}
    for d in lines[1:]:
        recs[d["key"]] = d
    qids = header["qids"]
    check(header["version"] == 3 and len(qids) == 18, "header names v3 and 18 questions")
    seeds = [d for d in recs.values() if d["gen"] == 0]
    check(len(seeds) == 24 and all(len(d["reps"]["0"]) == 18 and len(d["reps"]["1"]) == 18 for d in seeds),
          "every seed has both replicates on every question")
    gens = [json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
    check([g["gen"] for g in gens] == [0, 1, 2], "generations 0, 1, 2 logged")
    check(all(g["children"] <= 24 and g["parents"] == 12 for g in gens[1:]), "12 parents, at most 24 children",
          str([(g["parents"], g["children"]) for g in gens[1:]]))
    kids = [d for d in recs.values() if d["gen"] > 0]
    check(all(d["meta"]["slot"] in "AB" and d["meta"]["target"] in range(6) for d in kids),
          "every child records its slot and target group")
    check(all(d["op"] in P.EDIT_KINDS for d in kids), "every child is one named edit")
    check(all(recs[d["parent"]]["lineage"] == d["lineage"] for d in kids), "children inherit their seed lineage")
    failed = [d for d in kids if not d["meta"]["screen"]]
    passed = [d for d in kids if d["meta"]["screen"]]
    check(all(len(d["reps"]["0"]) == 18 for d in passed), "a child that passes is scored on every question")
    check(all(d["meta"]["d"] >= d["meta"]["se"] > -1 for d in failed)
          and all(d["meta"]["d"] <= d["meta"]["se"] for d in passed),
          "a child fails only when more than one standard error behind", f"{len(failed)} failed")
    # slot rules, recomputed from the archive on file
    s = V.Search(types.SimpleNamespace(out=str(TMP / "v3" / "check"), seed=0, max_calls_per_question=16),
                 {}, P.load_groups(CLUSTERS_V3, 3), None, {})
    for d in recs.values():
        r = P.ProgRecord.from_json(d)
        r.dup_of = None
        s.archive[r.key] = r
    for r in s.archive.values():
        s.mark_duplicate(r)
    slots = s.compute_slots()
    logged = {(x["group"], x["slot"]): x["key"] for x in gens[-1]["slots"]}
    check(slots == logged, "the logged slots follow from the archive")
    pool = s.eligible()
    check(all(abs(s.score(s.archive[slots[(g, "A")]], g) - max(s.score(r, g) for r in pool)) < 1e-9
              for g in s.group_ids), "slot A is the group's best score")
    check(all(slots[(g, "A")] != slots[(g, "B")] for g in s.group_ids), "slot B is never slot A's holder")
    check(all(abs(s.gaps_b[(g, slots[(g, "B")])] - max(s.gaps_b[(g, r.key)] for r in pool
                                                     if r.key != slots[(g, "A")])) < 1e-9
              for g in s.group_ids), "slot B is the largest rank gap")
    check(all(not s.archive[k].gaps(s.gq[g], 1) for (g, _), k in slots.items()),
          "every slot holder has both replicates on its group")
    check(all(s.archive[k].dup_of is None and not s.archive[k].gaps(s.qids, 0) for k in slots.values()),
          "slot holders are fully scored and not duplicates")
    summary = json.loads((run / "summary.json").read_text())
    check(sum(summary["holders_by_source"].values()) == 12
          and set(summary["holders_by_source"]) <= {"protocol", "llm", "random"},
          "the summary traces every slot holder to a seed source", str(summary["holders_by_source"]))

    # resume
    before = T.MODEL_CALLS["chat"]
    run_cli(V, common + ["--generations", "3", "--resume"])
    gens = [json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
    check(gens[-1]["gen"] == 3 and T.MODEL_CALLS["chat"] > before, "resume runs generation 3")
    before = T.MODEL_CALLS["chat"]
    run_cli(V, common + ["--generations", "3", "--resume"])
    check(T.MODEL_CALLS["chat"] == before, "a finished search resumes without spending")
    try:
        run_cli(V, [a if a != "3" else "4" for a in common] + ["--generations", "3", "--resume"])
        check(False, "a different question set is refused on resume")
    except SystemExit:
        check(True, "a different question set is refused on resume")

    # champions
    run_cli(V, common + ["--pick-champions", "--heldout-cap", "4"])
    champs = json.loads((run / "champions.json").read_text())
    check(champs["baselines"] == ["direct", "mad", "self_refine"], "baselines are named")
    ok = True
    for g, res in list(champs["per_group"].items()) + [("all", champs["global"])]:
        names = {p["name"] for p in res["programs"]}
        fin = [p for p in res["programs"] if p["finalist"]]
        ok &= {"direct", "mad", "self_refine"} <= names and 1 <= len(fin) <= 5
        ok &= res["champion"] in {p["key"] for p in fin}
        champ = next(p for p in res["programs"] if p["key"] == res["champion"])
        cheap = next(p for p in res["programs"] if p["key"] == res["cheapest_within_one_se"])
        ok &= cheap["vs_champion"]["within_one_se"] and (cheap["turns"] or 0) <= (champ["turns"] or 0)
        ok &= all(p["n_missing"] == 0 for p in res["programs"])
    check(ok, "each group: baselines scored, champion a finalist, cheapest within one SE no dearer")
    held = {q for c in json.loads(CLUSTERS_V3.read_text())["clusters"] for q in c["held_out"] + c["subset"][3:]}
    used = {q for res in champs["per_group"].values() for p in res["programs"] for q in p["marks"]}
    check(used <= held and not used & set(qids), "held-out questions are not search questions")
    after = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    check(len(after) == len([json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()])
          and all(set(d["reps"]["0"]) <= set(qids) for d in after[1:]),
          "held-out results never enter the search archive")


def test_interruptions(seeds_path: Path):
    """A run stopped part-way (here by the call cap, which takes the same exit
    as Ctrl-C) must resume into a complete, consistent archive."""
    print("interrupted runs")
    run, cache = TMP / "v3b" / "run", TMP / "v3b" / "rounds.jsonl"
    common = ["--seeds", str(seeds_path), "--out", str(run), "--clusters", str(CLUSTERS_V3),
              "--per-group", "3", "--live-cache", str(cache), "--workers", "8"]

    def state():
        lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
        recs = {d["key"]: d for d in lines if not d.get("header")}
        gens = ([json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
                if (run / "generations.jsonl").exists() else [])
        return recs, gens

    # stopped in the middle of generation 0
    run_cli(V, common + ["--generations", "1", "--max-total-calls", "400"])
    recs, gens = state()
    check(len(recs) == 0 and not gens, "a stop inside generation 0 stores no half-scored seed")
    run_cli(V, common + ["--generations", "0", "--resume"])
    recs, gens = state()
    seeds = [d for d in recs.values() if d["gen"] == 0]
    check(len(seeds) == 24 and all(len(d["reps"]["0"]) == 18 and len(d["reps"]["1"]) == 18 for d in seeds)
          and [g["gen"] for g in gens] == [0], "resume completes the seeds")

    # stopped in the middle of generation 1
    run_cli(V, common + ["--generations", "2", "--resume", "--max-total-calls", "120"])
    recs, gens = state()
    check([g["gen"] for g in gens] == [0], "a stop inside generation 1 does not log it as done",
          str([g["gen"] for g in gens]))
    n_partial_gen = sum(1 for d in recs.values() if d["gen"] == 1)
    run_cli(V, common + ["--generations", "2", "--resume"])
    recs, gens = state()
    check([g["gen"] for g in gens] == [0, 1, 2], "resume redoes generation 1 and goes on to 2",
          str([g["gen"] for g in gens]))
    kids1 = [d for d in recs.values() if d["gen"] == 1]
    check(len(kids1) >= n_partial_gen and len({d["name"] for d in recs.values()}) == len(recs),
          "children of the interrupted generation are kept and every name is unique",
          f"{n_partial_gen} before, {len(kids1)} after")
    holders = {x["key"]: x["group"] for x in gens[-1]["slots"]}
    groups = P.load_groups(CLUSTERS_V3, 3)
    gq = {g["group"]: g["search"] for g in groups["groups"]}
    check(all(set(gq[g]) <= set(recs[k]["reps"].get("1", {})) for k, g in holders.items()),
          "after the resume every slot holder is confirmed")
    summary = json.loads((run / "summary.json").read_text())
    check(summary["generations"] == 2 and len(summary["slots"]) == 12, "summary written after the resume")


if __name__ == "__main__":
    test_statistics()
    test_edits()
    test_subsets()
    seeds_path = test_seeds()
    test_search(seeds_path)
    test_interruptions(seeds_path)
    print()
    if T.FAILURES:
        print(f"{len(T.FAILURES)} FAILED: " + "; ".join(T.FAILURES))
        sys.exit(1)
    print(f"all checks passed (model calls: {T.MODEL_CALLS}); temp dir {TMP}")

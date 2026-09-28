"""Offline checks for the per-group search (program_space, program_guide,
program_seeds, evolve_program_clusters). The debate model is a fake OpenAI
client that answers with long replies and real summary follow-ups, so the v2
executor, the round cache and its keys run exactly as they would live; the
guide model is a fake that returns grammar-valid edits of the requested kind.
No vLLM, no API key, no cost.

    python scripts/test_program_clusters.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import openai  # noqa: E402

import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import schema_fitness as SF  # noqa: E402
import program_guide as G  # noqa: E402
import program_space as P  # noqa: E402

FAILURES: list[str] = []
MODEL_CALLS = {"chat": 0, "summary": 0}


def check(okay: bool, label: str, extra: str = "") -> None:
    print(("  ok   " if okay else "  FAIL ") + label + (f"  [{extra}]" if extra else ""))
    if not okay:
        FAILURES.append(label)


# --- fake debate model ---------------------------------------------------------------

class _Msg:
    def __init__(self, content):
        self.content, self.reasoning_content = content, None


class _Choice:
    def __init__(self, content):
        self.message, self.finish_reason = _Msg(content), "stop"


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeCompletions:
    """Answers deterministically from a hash of the conversation, with a
    per-process salt so replicates differ. A long reasoning reply (over
    debate_mcq.V2_SHORT_CHARS) forces the v2 summary call; the summary repeats
    the reply's letter."""
    salt = 0

    def create(self, model, messages, temperature, max_tokens, extra_body=None, **kw):
        last = messages[-1]["content"]
        if last == D.SUMMARY_NUDGE:
            MODEL_CALLS["summary"] += 1
            prior = messages[-2]["content"]
            letter = D.extract_letter(prior, 10) or "A"
            return _Resp(f"In short: the decisive fact points one way.\nANSWER: {letter}")
        MODEL_CALLS["chat"] += 1
        FakeCompletions.salt += 1
        seed = f"{messages[0]['content'][:30]}|{last}|{FakeCompletions.salt}"
        letter = "ABCD"[int(hashlib.sha1(seed.encode()).hexdigest(), 16) % 4]
        if "SURVIVING" in messages[0]["content"]:
            return _Resp("Ruling out. SURVIVING: A B")
        body = ("Let me work through the options carefully. " * 40).strip()
        return _Resp(f"{body}\nANSWER: {letter}")


class FakeChat:
    completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self, *a, **kw):
        self.chat = FakeChat()


openai.OpenAI = FakeOpenAI


# --- fake guide model ----------------------------------------------------------------

class FakeParsed:
    def __init__(self, parsed):
        self.output_parsed, self.status, self.usage = parsed, "completed", None


class FakeResponses:
    def __init__(self, rng):
        self.rng = rng
        self.calls = 0

    def parse(self, model, instructions, input, reasoning, text_format, max_output_tokens):
        self.calls += 1
        if text_format is G.SeedBatch:
            progs = []
            for i, (name, base) in enumerate(list(P.PROTOCOLS.items())[:4]):
                child, _ = P.mutate(base, self.rng, "plan")
                progs.append(G.SeedProgram(name=f"fake_{i}", strategy="a fake variant",
                                           program=G.Program(**child)))
            return FakeParsed(G.SeedBatch(programs=progs))
        parent = json.loads(input.split("CURRENT PROGRAM\n", 1)[1].split("\n\nSCORES BY GROUP")[0])
        req = re.search(r"REQUESTED EDIT: (.*)\.$", input, re.S).group(1)
        op = next(k for k, v in G.OP_TEXT.items() if v == req)
        child, _ = P.mutate(parent, self.rng, op)
        return FakeParsed(G.Edit(rationale="fake edit", program=G.Program(**child)))


class FakeGuide:
    def __init__(self):
        self.responses = FakeResponses(random.Random(5))


FAKE_GUIDE = FakeGuide()
G.make_client = lambda: FAKE_GUIDE

TMP = Path(tempfile.mkdtemp(prefix="cluster_search_test_"))
CLUSTERS = P.ROOT / "outputs/clusters_train_both.json"
DATASET = P.ROOT / "datasets/supergpqa_program_search_train.json"
CACHE = TMP / "rounds.jsonl"


# --- unit checks ---------------------------------------------------------------------

def test_space():
    print("program space")
    settings = P.configure_executor()
    check(D.V2 and D.digest_signature() == (300, 900), "executor is v2 with the 300+900 window")
    check("eliminate" not in M.ACTIONS and "elim_blind" not in M.ACTIONS, "eliminator moves dropped")
    key = SF.path_key([P.solvers(2)], {"critic": "x"})
    check('"v":"2"' in key and '"d":[300,900]' in key and '"p":' in key,
          "cache key carries v2, the window and the prompt hash", key)
    for name, prog in P.PROTOCOLS.items():
        try:
            P.validate_program(prog)
            ok = True
        except Exception as exc:
            ok, name = False, f"{name}: {exc}"
        check(ok, f"protocol {name} validates")
    check(len(P.PROTOCOLS) == 8, "eight protocols")
    check(all(k not in P.PROTOCOLS for k in ("program_b", "spend_everything")), "no program_b, no spend-everything")
    rng = random.Random(1)
    n_ok = 0
    for _ in range(300):
        p = P.random_program(rng)
        try:
            P.validate_program(p)
            n_ok += 1
        except Exception:
            pass
    check(n_ok == 300, "300 random programs validate")
    base = P.PROTOCOLS["early_exit_agree"]
    ops = set()
    for _ in range(200):
        child, op = P.mutate(base, rng)
        P.validate_program(child)
        check_ok = P.canon(child) != P.canon(base)
        if not check_ok:
            check(False, "mutation changed the program")
        ops.add(P.op_family(op))
        want = P.expected_rule_delta(op)
        got = len(child["rules"]) - len(base["rules"])
        if want is not None and got != want:
            check(False, f"rule delta {got} for {op}")
    check("plan" in ops and "drop_rule" in ops and "add_rule" in ops, "mutation covers plan and rule ops", str(sorted(ops)))
    a, b = P.struct_tokens(P.PROTOCOLS["direct"]), P.struct_tokens(P.PROTOCOLS["mad"])
    check(P.struct_distance(a, a) == 0.0 and 0 < P.struct_distance(a, b) <= 1, "structural distance")
    picked = P.farthest_point([a, b, a], 2, P.struct_distance, fixed=[a])
    check(picked[0] == 1, "farthest-point picks the different one first", str(picked))
    g3, g5 = P.load_groups(CLUSTERS, 3), P.load_groups(CLUSTERS, 5)
    check(all(x["search"] == y["search"][:3] for x, y in zip(g3["groups"], g5["groups"])),
          "search sets are prefixes across --per-group")
    check(all(set(x["search"]).isdisjoint(x["held_out"]) for x in g5["groups"]), "held-out disjoint from search")
    check(all(len(x["search"]) + len(x["held_out"]) == x["size"] for x in g5["groups"]),
          "search + held-out = group size")
    txt = P.grammar_text()
    check("eliminat" not in txt and "pair_expert_solver" in txt, "grammar text matches the live menu")
    return settings


def run_cli(module, argv: list[str]) -> None:
    old = sys.argv
    sys.argv = [module.__name__] + argv
    try:
        module.main()
    finally:
        sys.argv = old


def test_seeds():
    print("seed stage")
    import program_seeds
    out = TMP / "seeds.json"
    run_cli(program_seeds, ["--per-group", "3", "--random-pool", "20", "--n-random", "4",
                            "--sanity-n", "4", "--workers", "4", "--out", str(out),
                            "--live-cache", str(CACHE), "--n-llm", "4"])
    seeds = json.loads(out.read_text())
    srcs = [s["source"] for s in seeds["seeds"]]
    check(srcs.count("protocol") == 8 and srcs.count("llm") == 4 and srcs.count("random") == 4,
          "8 + 4 + 4 seeds (test sizes)", str(srcs))
    check(all(2 <= s["sanity"]["mean_turns"] <= 12 for s in seeds["seeds"] if s["source"] == "random"),
          "random seeds passed the sanity window")
    check(len({P.canon(s["program"]) for s in seeds["seeds"]}) == len(seeds["seeds"]), "seeds are distinct")
    check(out.with_suffix(".md").exists() and out.with_suffix(".llm.json").exists(), "report and llm batch written")
    lines = [json.loads(l) for l in CACHE.open() if l.strip()]
    check(lines and all('"v":"2"' in r["k"] for r in lines), "every recording carries the v2 key")
    check(any(D.SUMMARY_MARK in t for r in lines for _, t in r["responses"]),
          "recordings contain summary follow-ups")
    check(MODEL_CALLS["summary"] > 0, "summary calls were made", str(MODEL_CALLS))
    return out


def test_search(seeds_path: Path):
    print("search")
    import evolve_program_clusters as E
    run = TMP / "run"
    common = ["--per-group", "3", "--children", "6", "--immigrants", "1", "--elite-slots", "2",
              "--lexicase-slots", "2", "--diversity-slots", "1", "--workers", "4",
              "--novelty-budget", "60", "--live-cache", str(CACHE), "--out", str(run),
              "--seeds", str(seeds_path), "--print-every", "1", "--rare-k", "2"]
    run_cli(E, common + ["--generations", "2"])
    archive = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    header = archive[0]
    recs = {}
    for d in archive[1:]:
        recs[d["key"]] = d
    check(header.get("header") and header["per_group"] == 3 and header["settings"]["v2"], "archive header")
    check(len(recs) >= 16 + 6, "seeds and children in the archive", str(len(recs)))
    seeds = [d for d in recs.values() if d["gen"] == 0]
    check(all(len(d["reps"]["0"]) == 18 for d in seeds), "every seed covered all 18 search questions")
    check(any("1" in d["reps"] for d in recs.values()), "some programs have replicate-1 results")
    gens = [json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
    check([g["gen"] for g in gens] == [0, 1, 2], "generation log 0..2", str([g["gen"] for g in gens]))
    check(all(g.get("children", 0) >= 1 for g in gens[1:]), "each generation produced children")
    # on 18 questions most edits are neutral, so a guided child may not survive
    # every run; what must hold is that guided edits were attempted and either
    # became children or fell back for a named reason
    tried = sum(g.get("guided_tried", 0) for g in gens[1:])
    made = sum(g.get("guided_child", 0) for g in gens[1:])
    fell = sum(g.get("guided_fallback", 0) for g in gens[1:])
    check(tried >= 1 and made + fell <= tried, "guided edits were attempted",
          f"tried {tried}, children {made}, fallbacks {fell}")
    check(FAKE_GUIDE.responses.calls >= 2, "guide model was called", str(FAKE_GUIDE.responses.calls))
    summary = json.loads((run / "summary.json").read_text())
    check(len(summary["cells"]) >= 6, "cells occupied", str(len(summary["cells"])))
    kids = [d for d in recs.values() if d["gen"] > 0 and d["op"] != "immigrant"]
    check(all(d["parent"] in recs for d in kids), "children point at archived parents")
    check(any(d["op"] == "immigrant" for d in recs.values()), "immigrants in the archive")
    # behaviour identity: among selectable programs no two share (path, letter)
    # on all questions; a program marked dup_of matches its owner
    beh = {}
    dup = bad_owner = 0
    for d in recs.values():
        r0 = d["reps"]["0"]
        if len(r0) == 18:
            h = P.behaviour_hash(r0)
            if d.get("dup_of"):
                bad_owner += P.behaviour_hash(recs[d["dup_of"]]["reps"]["0"]) != h
                continue
            dup += h in beh
            beh[h] = d["key"]
    check(dup == 0 and bad_owner == 0, "no behavioural duplicates among selectable programs",
          f"dup {dup}, bad owner {bad_owner}")
    check(all(d["op"] != "immigrant" or not d.get("dup_of") for d in recs.values()),
          "duplicate immigrants are not stored")

    # resume for one more generation
    calls_before = MODEL_CALLS["chat"]
    run_cli(E, common + ["--generations", "3", "--resume"])
    gens = [json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
    check(gens[-1]["gen"] == 3, "resume continued to generation 3", str(gens[-1]["gen"]))
    check(MODEL_CALLS["chat"] > calls_before, "resume spent new calls only on new work")

    # widen to 4 per group: seeds and elites get the new questions, header updated
    run_cli(E, [a if a != "3" else "4" for a in common] + ["--generations", "3", "--resume"])
    archive = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    headers = [d for d in archive if d.get("header")]
    check(len(headers) == 2 and headers[-1]["per_group"] == 4, "widened header appended")
    recs = {}
    for d in archive:
        if not d.get("header"):
            recs[d["key"]] = d
    seeds = [d for d in recs.values() if d["gen"] == 0]
    check(all(len(d["reps"]["0"]) == 24 for d in seeds), "seeds cover all 24 after widening")

    # shrinking is refused
    try:
        run_cli(E, common + ["--generations", "3", "--resume"])
        check(False, "shrinking per-group is refused")
    except SystemExit as exc:
        check("can only grow" in str(exc), "shrinking per-group is refused", str(exc)[:60])

    # champions
    run_cli(E, [a if a != "3" else "4" for a in common] + ["--pick-champions", "--heldout-cap", "3", "--reps", "2"])
    champ = json.loads((run / "champions.json").read_text())
    check(len(champ["per_group"]) == 6 and all(v["champion"] for v in champ["per_group"].values()),
          "a champion per group")
    check(champ["global"]["champion"] is not None and champ["global"]["held_out_n"] == 18, "global champion on 18 held-out")
    fin = champ["per_group"]["0"]["finalists"][0]
    check(set(fin["per_rep"]) == {"0", "1"} and all(v["n"] == 3 for v in fin["per_rep"].values()),
          "finalists ran two replicates on the held-out sample")
    check(all(q not in header["qids"] for q in fin["per_rep"]["0"]["marks"]), "held-out questions are not search questions")


def test_search_internals(seeds_path: Path):
    """Direct checks on Search methods, on the archive the CLI run left."""
    print("search internals")
    import evolve_program_clusters as E
    import argparse
    run = TMP / "run"
    ns = argparse.Namespace(out=run, cost_bands="4,8", seed=0, max_calls_per_question=16,
                            workers=4, novelty_budget=60, prescreen_margin=0.02, top_n=5,
                            rare_k=2, topup_window=5, children=6, immigrants=1, elite_slots=2,
                            lexicase_slots=2, diversity_slots=1, floor=0.08, guided_frac=0.0,
                            plan_prob=0.2, neutral_retries=1, immigrant_pool=5, n_fail_digests=3,
                            n_ok_digests=1, guide_effort="medium", print_every=1)
    rows = {r["id"]: r for r in json.loads(DATASET.read_text())}
    groups = P.load_groups(CLUSTERS, 4)
    settings = dict(P.configure_executor(), model=P.DEFAULT_MODEL)
    runner = P.make_runner(rows, CACHE, model=P.DEFAULT_MODEL, lock=True)
    try:
        s = E.Search(ns, rows, groups, runner, None, random.Random(0), settings)
        s.open_archive(resume=True)
        # interleaving: consecutive questions come from different groups
        order = s.interleaved(list(s.qids))
        firsts = [s.group_of[q] for q in order[:6]]
        check(len(set(firsts)) == 6, "interleaved fill order rotates over the groups", str(firsts))
        # seed recovery: drop one seed from the loaded archive and ask for it back
        seeds = json.loads(seeds_path.read_text())["seeds"]
        victim = P.canon(P.normalize_program(seeds[0]["program"]))
        del s.archive[victim]
        n = s.ensure_seeds(seeds)
        check(n == 1 and victim in s.archive and not s.archive[victim].gaps(s.qids, 0),
              "a missing seed is re-added and fully covered on resume", str(n))
        # prescreen arithmetic on a synthetic partial record
        s.recompute_cells()
        g0 = s.group_ids[0]
        elite = s.archive[s.cells[(g0, 0)]] if (g0, 0) in s.cells else None
        rec = E.ProgRecord(P.PROTOCOLS["direct"], "synthetic", "synthetic", 99)
        check(s.passes_screen(rec, 1.0) == (True, "first look"), "an uncovered program gets a first look")
        q = s.gq[g0][0]
        rec.record(0, q, {"correct": True, "n_calls": 1, "letter": "A", "actions": ["continue"]})
        ok, why = s.passes_screen(rec, 1.0)
        check(ok, "one correct answer projects to full marks and passes", why)
        rec.record(0, q, {"correct": False, "n_calls": 1, "letter": "A", "actions": ["continue"]})
        ok, why = s.passes_screen(rec, 1.0)
        floor_zero = elite is None or elite.score(s.gq[g0]) <= ns.prescreen_margin
        check(ok == floor_zero or ok is False, "a zero projection fails unless the elite is also at zero", why)
        # digest text for the guide
        seed_rec = next(r for r in s.live() if r.name == "mad")
        digests = G.build_digests(seed_rec.program, seed_rec, runner, rows, s.gq[g0], g0,
                                  random.Random(0), n_fail=2, n_ok=1)
        check(digests and all("correct letter" in d and "round 1 solver x3" in d for d in digests),
              "digests list rounds and the correct letter", digests[0][:80] if digests else "none")
        check(all("[v2 summary" not in d for d in digests), "digests show the summary text, not the marker")
        # lexicase returns a live program; combined distance is symmetric and bounded
        a, b = s.live()[0], s.live()[1]
        d1, d2 = P.combined_distance(a, b, s.qids), P.combined_distance(b, a, s.qids)
        check(abs(d1 - d2) < 1e-12 and 0 <= d1 <= 1, "combined distance symmetric and in [0,1]")
        check(s.lexicase(s.live()) is not None, "lexicase picks a parent")
    finally:
        runner.close()


def test_guide_checks():
    print("guided-edit checks")
    parent = P.PROTOCOLS["early_exit_agree"]

    class Client:
        class responses:
            @staticmethod
            def parse(**kw):
                child = P.copy_program(parent)
                child["rules"].append({"when": ["step==3"], "do": "critic"})   # an addition
                return FakeParsed(G.Edit(rationale="added", program=G.Program(**child)))

    child, why = G.guided_edit(Client(), parent, "drop_rule", [], "g", [])
    check(child is None and "rule count" in why, "an addition is rejected when a drop was requested", why)
    child, why = G.guided_edit(Client(), parent, "add_rule", [], "g", [])
    check(child is not None and len(child["rules"]) == len(parent["rules"]) + 1, "an addition is accepted when requested")

    class Same:
        class responses:
            @staticmethod
            def parse(**kw):
                return FakeParsed(G.Edit(rationale="none", program=G.Program(**parent)))

    child, why = G.guided_edit(Same(), parent, "change_action", [], "g", [])
    check(child is None and why == "unchanged", "an unchanged program is rejected")


if __name__ == "__main__":
    test_space()
    test_guide_checks()
    seeds_path = test_seeds()
    test_search(seeds_path)
    test_search_internals(seeds_path)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        sys.exit(1)
    print(f"all checks passed (model calls: {MODEL_CALLS}); temp dir {TMP}")

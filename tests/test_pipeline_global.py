"""Offline checks for the global pipeline: the final-read repair (--last-round-vote), the
train / dev split, the cost levels, the global-pipeline seed stage, the global search (one train question
set, three level slots, no second run, the dev split never run by the search, resume),
the final choice on the dev split, and the test evaluation. The debate model and the guide
model are the fakes of test_pipeline_cluster.py (no vLLM, no API key, no cost).

    python tests/test_pipeline_global.py
"""

from __future__ import annotations

import ast
import json
import random
import shutil
import sys
import types
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))   # the code under test
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # the other test files

import test_pipeline_cluster as T3  # noqa: E402  (installs the fake debate and guide models)
import test_program_clusters as T  # noqa: E402
import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_guide as G  # noqa: E402
import program_space as P  # noqa: E402
import gap_report as GR  # noqa: E402
import split_train_dev as SPL  # noqa: E402
import program_seeds_global as S4  # noqa: E402
import evolve_pipeline_global as V4  # noqa: E402
import eval_pipeline_global as E4  # noqa: E402

check, run_cli, TMP = T.check, T.run_cli, T.TMP
W = TMP / "v4"
WINDOW = ["--context-window", "65536"]
V4_EXEC = ["--high-cost", "3", "--turn-cap", "15", "--any-round-width", "--last-round-vote"] + WINDOW

# the fake guide learns the one-program-per-level request
_old_parse = T.FakeResponses.parse
LEVEL_REQUESTS: list[dict] = []


def _parse(self, model, instructions, input, reasoning, text_format, max_output_tokens):
    if text_format is G.LevelSeedBatch:
        self.calls += 1
        LEVEL_REQUESTS.append({"instructions": instructions, "input": input})
        need = ast.literal_eval(input.split("each of these cost levels: ", 1)[1].split("]", 1)[0] + "]")
        progs = []
        for lv in need:
            child, _ = P.mutate_uniform(P.PROTOCOLS["fresh_on_disagree"], self.rng)
            progs.append(G.LevelSeedProgram(level=lv, name=f"fake {lv}", strategy="a fake variant",
                                            program=G.Program(**child)))
        return T.FakeParsed(G.LevelSeedBatch(programs=progs))
    return _old_parse(self, model, instructions, input, reasoning, text_format, max_output_tokens)


T.FakeResponses.parse = _parse


# --- the final-read repair -------------------------------------------------------------------

class ScriptedRunner:
    """run_round returns the scripted round for the number of rounds run so far."""

    def __init__(self, rounds):
        self.rounds = rounds

    def run_round(self, qid, all_rounds, executed_specs, round_spec, rep=0, prompts=None):
        return self.rounds[len(all_rounds)]


def say(*letters):
    return [("solver", f"Reasoning.\nANSWER: {l}" if l else "No commitment here.") for l in letters]


def test_read_repair():
    print("final-read repair")
    row = {"id": "q1", "question": "Which?", "options": [f"o{i}" for i in range(10)], "answer_letter": "B"}
    four = {"personas": ["solver"] * 4}
    stop3 = lambda read: {"plan": [four], "rules": [P.CONT(), {"when": ["step==1", "r1_majority>=3"],
                                                              "do": f"stop:{read}"}], "default": "stop:vote"}
    runner = ScriptedRunner([say("B", "B", "B", "D")])
    got = {}
    for on in (False, True):
        P.configure_executor(executor="v3", window=65536, last_round_vote=on)
        for read in ("last_commit", "last_speaker"):
            got[(on, read)] = M.run_program(P.normalize_program(stop3(read)), runner, row)["letter"]
    check(got[(False, "last_commit")] == "D" and got[(False, "last_speaker")] == "D",
          "off: a 3-1 round ending in its dissenter reads the dissenter (every earlier run)", str(got))
    check(got[(True, "last_commit")] == "B" and got[(True, "last_speaker")] == "B",
          "on: the same round reads its majority", str(got))
    # one speaker per round (self-refine): the same answer either way, also when the last round commits nothing
    chain = {"plan": [{"personas": ["solver"]}, {"personas": ["critic"]}, {"personas": ["solver"]}],
             "rules": [P.CONT()], "default": "stop:last_commit"}
    for script, want in (([say("A"), [("critic", "Flawed.\nANSWER: C")], say("B")], "B"),
                         ([say("A"), [("critic", "Flawed.\nANSWER: C")], say(None)], "C")):
        outs = []
        for on in (False, True):
            P.configure_executor(executor="v3", window=65536, last_round_vote=on)
            outs.append(M.run_program(P.normalize_program(chain), ScriptedRunner(script), row)["letter"])
        check(outs == [want, want], f"a one-speaker last round reads the same either way ({want})", str(outs))
    last_sp = {**chain, "default": "stop:last_speaker"}
    outs = []
    for on in (False, True):
        P.configure_executor(executor="v3", window=65536, last_round_vote=on)
        outs.append(M.run_program(P.normalize_program(last_sp), ScriptedRunner(
            [say("A"), [("critic", "Flawed.\nANSWER: C")], say(None)]), row)["letter"])
    check(outs == [None, "C"], "last_speaker with a silent last speaker: no answer before, the last answer now",
          str(outs))
    check(M.last_round_vote([say("A", "B")], 10) == "A" and M.last_round_vote([say("C"), say(None, None)], 10) == "C"
          and M.last_round_vote([say(None)], 10) is None,
          "ties go to the earliest speaker; a round with no commitment is skipped")
    s_on = P.configure_executor(executor="v3", window=65536, last_round_vote=True)
    on = M.LAST_ROUND_VOTE
    s_off = P.configure_executor(executor="v3", window=65536)
    check(on and not M.LAST_ROUND_VOTE and s_on.get("last_round_vote") is True and "last_round_vote" not in s_off,
          "the setting is named only when on, and configuring again without it switches it off")
    try:
        P.configure_executor(executor="v2", last_round_vote=True)
        check(False, "the v2 executor refuses the repair")
    except SystemExit:
        check(True, "the v2 executor refuses the repair")
    P.configure_executor(executor="v3", window=65536)
    old = ("  stop:last_commit     stop; answer = the most recent committed letter\n"
           "  stop:last_speaker    stop; answer = the last round's letter\n"
           "  stop:vote            stop; answer = plurality over every letter committed so far\n")
    check(old in P.grammar_text(), "off: the grammar text the seed writer saw before is unchanged")
    P.configure_executor(executor="v3", window=65536, last_round_vote=True)
    txt = P.grammar_text()
    check("most common letter of the last round" in txt and "the most recent committed letter" not in txt,
          "on: the grammar text describes the repaired reads")
    P.configure_executor(executor="v3", window=65536)


def test_settings():
    print("settings")
    # settings as a run made now records them (run3's flags), built here: no earlier run is read
    run3 = {**P.configure_executor(executor="v3", window=32768, high_cost=3, turn_cap=20, judge_persona=True,
                                   any_round_width=True, visible_reasoning=True), "model": "openai/gpt-oss-20b"}
    try:
        GR.configure_like(run3)
        check(True, "run3's flags are reproduced exactly (no new key without the repair)")
    except SystemExit as exc:
        check(False, "run3's flags are reproduced exactly", str(exc))
    v4 = {k: v for k, v in run3.items() if k not in ("judge_persona", "prompts")} | {"last_round_vote": True}
    v4["prompts"] = P.configure_executor(executor="v3", window=32768, high_cost=3, turn_cap=20, any_round_width=True,
                                         visible_reasoning=True, last_round_vote=True)["prompts"]
    try:
        GR.configure_like(v4)
        check(M.LAST_ROUND_VOTE, "global-pipeline settings round-trip through the analysis tools, repair on")
    except SystemExit as exc:
        check(False, "global-pipeline settings round-trip through the analysis tools", str(exc))
    P.configure_executor(executor="v3", window=65536)


class _LabelGuide:
    """A guide that labels levels loosely, then names one it was asked for again."""

    def __init__(self, batches):
        self.batches, self.inputs = list(batches), []
        self.responses = self

    def parse(self, model, instructions, input, reasoning, text_format, max_output_tokens):
        self.inputs.append(input)
        rng = random.Random(len(self.inputs))
        progs = []
        for i, lv in enumerate(self.batches.pop(0)):
            while True:
                child, _ = P.mutate_uniform(P.PROTOCOLS["fresh_on_disagree"], rng)
                try:
                    G.to_program(G.Program(**child))
                    break
                except Exception:
                    continue
            progs.append(G.LevelSeedProgram(level=lv, name=f"p{len(self.inputs)}_{i}", strategy="x",
                                            program=G.Program(**child)))
        return T.FakeParsed(G.LevelSeedBatch(programs=progs))


def test_level_labels():
    print("level labels of the seed writer")
    guide = _LabelGuide([["Cheap", "medium level", "pricey"], ["EXPENSIVE"]])
    got = G.write_level_seeds(guide, {}, "  examples", V4.level_text(15), n_examples=1)
    check([w["level"] for w in got] == ["cheap", "medium", "expensive"] and len(guide.inputs) == 2,
          "loose labels name their level; an unknown one is asked for again", str([w["level"] for w in got]))
    check("'pricey' is not one of" in guide.inputs[1] and "['expensive']" in guide.inputs[1],
          "the second request names the bad label and asks only for the missing level")


class _ScriptGuide:
    """A guide that returns scripted (label, program) batches; an Exception in place of a
    batch is raised (a failed call)."""

    def __init__(self, batches):
        self.batches, self.inputs, self.instructions = list(batches), [], []
        self.responses = self

    def parse(self, model, instructions, input, reasoning, text_format, max_output_tokens):
        self.inputs.append(input)
        self.instructions.append(instructions)
        batch = self.batches.pop(0)
        if isinstance(batch, Exception):
            raise batch
        return T.FakeParsed(G.LevelSeedBatch(programs=[
            G.LevelSeedProgram(level=lv, name=f"p{len(self.inputs)}_{i}", strategy="x", program=G.Program(**prog))
            for i, (lv, prog) in enumerate(batch)]))


def _prog(plan: list[dict], rules: list[dict], default: str = "stop:vote") -> dict:
    return {"plan": plan, "rules": rules, "default": default}


def test_seed_writer():
    print("the seed writer")
    P.configure_executor(executor="v3", window=65536, high_cost=3, turn_cap=15, any_round_width=True)
    hi = {"effort": "high"}
    cheap = _prog([{"personas": ["solver", "solver"]}], [P.CONT(), {"when": ["step==1"], "do": "solver|blind"}])
    medium = _prog([{"personas": ["solver", "solver"], **hi}], [P.CONT()])
    expensive = _prog([{"personas": ["solver"] * 4, **hi}], [P.CONT()])
    over_cap = _prog([{"personas": ["solver"] * 4, **hi}, {"personas": ["solver"] * 3, **hi}], [P.CONT()])
    old_sleep = G.time.sleep
    G.time.sleep = lambda s: None
    old_repair = M.LAST_ROUND_VOTE
    try:
        guide = _ScriptGuide([[("level: cheap", cheap), ("medium-cost", medium), ("Expensive.", expensive)]])
        got = G.write_level_seeds(guide, {}, "  examples", V4.level_text(15), n_examples=1)
        check([w["level"] for w in got] == ["cheap", "medium", "expensive"] and len(guide.inputs) == 1,
              "labels are read by word ('level: cheap', 'medium-cost', 'Expensive.')", str([w["level"] for w in got]))
        guide = _ScriptGuide([[("cheap", cheap), ("medium", medium), ("expensive", over_cap)],
                              [("expensive", expensive)]])
        got = G.write_level_seeds(guide, {}, "  examples", V4.level_text(15), n_examples=1)
        check([w["level"] for w in got] == ["cheap", "medium", "expensive"] and "over the 15-turn cap" in guide.inputs[1],
              "a plan over the turn cap is refused and asked for again", guide.inputs[-1][-300:])
        check("these 1 cost levels" in guide.instructions[1] and "  cheap" not in guide.instructions[1]
              and "Return 1 programs" in guide.instructions[1] and "for the group" not in guide.instructions[0],
              "a retry's instructions name only the missing level; no v3 group text")
        M.set_last_round_vote(True)
        speaker = dict(P.PROTOCOLS["direct_high"], default="stop:last_speaker")
        guide = _ScriptGuide([[("cheap", speaker)], [("cheap", cheap)]])
        got = G.write_level_seeds(guide, {"direct_high": P.PROTOCOLS["direct_high"]}, "  examples",
                                  V4.level_text(15)[:1], n_examples=1)
        check(len(guide.inputs) == 2 and "identical to an existing program" in guide.inputs[1]
              and got[0]["program"] == P.normalize_program(cheap),
              "with the repair, a copy that differs only in last_speaker / last_commit is refused")
        M.set_last_round_vote(old_repair)
        # a failed call after a paid reply: what was accepted is handed out before the error
        kept = []
        guide = _ScriptGuide([[("cheap", cheap), ("medium", medium), ("expensive", over_cap)]]
                             + [RuntimeError("server error")] * 3)
        try:
            G.write_level_seeds(guide, {}, "  examples", V4.level_text(15), n_examples=1,
                                on_reply=lambda progs: kept.append(list(progs)))
            check(False, "a failed call raises")
        except RuntimeError:
            check(kept and [w["level"] for w in kept[-1]] == ["cheap", "medium"],
                  "the programs of a paid reply are handed out before a later call fails")
        guide = _ScriptGuide([[("expensive", expensive)]])
        got = G.write_level_seeds(guide, {}, "  examples", V4.level_text(15), n_examples=1, written=kept[-1])
        check([w["level"] for w in got] == ["cheap", "medium", "expensive"] and len(guide.inputs) == 1
              and "['expensive']" in guide.inputs[0],
              "a continued run asks only for the missing level")
    finally:
        G.time.sleep = old_sleep
        M.set_last_round_vote(old_repair)


def test_levels_and_splits():
    print("levels and splits")
    check([V4.level_of(t) for t in (0, 3, 4.5, 4.505, 9, 10.5, 10.51, 20)]
          == ["cheap", "cheap", "cheap", "medium", "medium", "medium", "expensive", "expensive"],
          "level limits: cheap <= 4.5 < medium <= 10.5 < expensive")
    check(V4.level_text(15) == [("cheap", "up to 4.5 turns"), ("medium", "more than 4.5, up to 10.5 turns"),
                                ("expensive", "more than 10.5 turns (a question is capped at 15)")],
          "the level descriptions", str(V4.level_text(15)))
    ids = [f"q{i:02d}" for i in range(30)]
    a, b = SPL.make_split(ids, 10, 0), SPL.make_split(ids, 10, 0)
    check(a == b and len(a["dev"]) == 10 and len(a["train"]) == 20 and not set(a["dev"]) & set(a["train"])
          and sorted(a["dev"] + a["train"]) == ids and a["train"] == [q for q in ids if q in a["train"]],
          "a split is a fixed draw: disjoint, complete, in the dataset's order")
    check(SPL.make_split(ids, 10, 1)["dev"] != a["dev"], "another seed draws another split")
    data = W / "split_data.json"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps([{"id": q} for q in ids]))
    out = W / "split.json"
    run_cli(SPL, ["--dataset", str(data), "--n-dev", "10", "--seed", "0", "--out", str(out)])
    run_cli(SPL, ["--dataset", str(data), "--n-dev", "10", "--seed", "0", "--out", str(out)])
    check(SPL.load_split(out)["dev"] == a["dev"], "the split file is written once and kept on a rerun")
    try:
        run_cli(SPL, ["--dataset", str(data), "--n-dev", "10", "--seed", "3", "--out", str(out)])
        check(False, "a different split over an existing file is refused")
    except SystemExit:
        check(True, "a different split over an existing file is refused")
    bad = W / "bad_split.json"
    bad.write_text(json.dumps({"train": ["a", "b"], "dev": ["b"]}))
    try:
        SPL.load_split(bad)
        check(False, "a split whose lists overlap is refused")
    except SystemExit:
        check(True, "a split whose lists overlap is refused")


# --- the toy run ---------------------------------------------------------------------------

def toy_data() -> tuple[Path, Path, Path, list[str]]:
    """A 24-question train file split 16 train / 8 dev, and a 10-question test file."""
    rows = json.loads(T.DATASET.read_text())
    rng = random.Random(7)
    picked = rng.sample(rows, 34)
    train_file, test_file = W / "toy_train.json", W / "toy_test.json"
    train_file.write_text(json.dumps(picked[:24]))
    test_file.write_text(json.dumps(picked[24:]))
    splits = W / "run" / "splits.json"
    run_cli(SPL, ["--dataset", str(train_file), "--n-dev", "8", "--seed", "0", "--out", str(splits)])
    return train_file, test_file, splits, [r["id"] for r in picked[24:]]


def test_seeds(train_file: Path, splits: Path) -> Path:
    print("global-pipeline seed stage")
    out = W / "run" / "seeds.json"
    argv = ["--splits", str(splits), "--dataset", str(train_file), "--out", str(out), "--seed-examples", "5"] + V4_EXEC
    n0 = len(LEVEL_REQUESTS)
    run_cli(S4, argv)
    seeds = json.loads(out.read_text())
    src = Counter(s["source"] for s in seeds["seeds"])
    check(src == {"protocol": 8, "llm": 3}, "8 literature seeds and 3 model-written ones", str(dict(src)))
    check(sorted(s["level"] for s in seeds["seeds"] if s["source"] == "llm") == ["cheap", "expensive", "medium"],
          "one model-written seed per cost level")
    check(not any(s["source"] == "judge" for s in seeds["seeds"]), "no judge seeds without the judge speaker")
    split = SPL.load_split(splits)
    req = LEVEL_REQUESTS[-1]
    check(len(LEVEL_REQUESTS) == n0 + 1 and len(seeds["examples"]) == 5 and set(seeds["examples"]) <= set(split["train"]),
          "the guide model is asked once and shown 5 train questions")
    rows = {r["id"]: r for r in json.loads(train_file.read_text())}
    check(not any(rows[q]["question"][:60] in req["input"] for q in split["dev"]), "no dev question is shown")
    check("difficulty" not in req["input"] and "answer_letter" not in req["input"]
          and all(n in req["instructions"] for n in ("cheap", "medium", "expensive")),
          "the levels are named; no difficulty and no answers are shown")
    check("most common letter of the last round" in req["instructions"],
          "the seed writer sees the repaired stop reads")
    check(seeds["settings"].get("last_round_vote") is True and "judge_persona" not in seeds["settings"],
          "the seeds record the global-pipeline settings")
    check(all(not P.first_action(s["program"]).startswith("stop:") for s in seeds["seeds"]), "every seed starts its debate")
    # the written programs are cached: a redo asks nothing
    out.rename(W / "run" / "seeds_first.json")
    run_cli(S4, argv)
    check(len(LEVEL_REQUESTS) == n0 + 1 and json.loads(out.read_text())["seeds"] == seeds["seeds"],
          "a redo reuses the cached model-written programs")
    try:
        run_cli(S4, argv)
        check(False, "an existing seed file is never overwritten")
    except SystemExit:
        check(True, "an existing seed file is never overwritten")
    cached = json.loads(out.with_suffix(".llm.json").read_text())
    for label, bad in (("an example that is not a train question", {**cached, "examples": cached["examples"][:-1]
                                                                       + [split["dev"][0]]}),
                       ("other settings", {**cached, "settings": {**cached["settings"], "turn_cap": 16}})):
        other = W / "run" / f"seeds_bad_{len(label)}.json"
        other.with_suffix(".llm.json").write_text(json.dumps(bad))
        try:
            run_cli(S4, [a if a != str(out) else str(other) for a in argv])
            check(False, f"a cache of written seeds with {label} is refused")
        except SystemExit:
            check(True, f"a cache of written seeds with {label} is refused")
    # a call that fails after a paid reply: the reply is on file, the run goes on later
    hi = {"effort": "high"}
    cheap = _prog([{"personas": ["solver", "solver"]}], [P.CONT(), {"when": ["step==1"], "do": "solver|blind"}])
    medium = _prog([{"personas": ["solver", "solver"], **hi}], [P.CONT(), {"when": ["step==1"], "do": "critic"}])
    expensive = _prog([{"personas": ["solver"] * 4, **hi}], [P.CONT(), {"when": ["step==1"], "do": "critic|high"}])
    bad = _prog([{"personas": ["solver"]}], [{"when": ["plan_left"], "do": "judge"}])
    fail_out = W / "run" / "seeds_fail.json"
    old_client, old_sleep = G.make_client, G.time.sleep
    G.time.sleep = lambda s: None
    try:
        first = _ScriptGuide([[("cheap", cheap), ("medium", medium), ("expensive", bad)]] + [RuntimeError("down")] * 3)
        G.make_client = lambda: first
        try:
            run_cli(S4, [a if a != str(out) else str(fail_out) for a in argv])
            check(False, "a failed call stops the seed stage")
        except RuntimeError:
            cached = json.loads(fail_out.with_suffix(".llm.json").read_text())
            check(not fail_out.exists() and cached["done"] is False
                  and [w["level"] for w in cached["programs"]] == ["cheap", "medium"],
                  "a failed call stops the seed stage with the paid reply's programs on file")
        second = _ScriptGuide([[("expensive", expensive)]])
        G.make_client = lambda: second
        run_cli(S4, [a if a != str(out) else str(fail_out) for a in argv])
        got = json.loads(fail_out.read_text())
        check(len(second.inputs) == 1 and "['expensive']" in second.inputs[0]
              and sorted(s["level"] for s in got["seeds"] if s["source"] == "llm") == ["cheap", "expensive", "medium"]
              and json.loads(fail_out.with_suffix(".llm.json").read_text())["done"] is True,
              "the next run asks only for the missing level and finishes the seeds")
    finally:
        G.make_client, G.time.sleep = old_client, old_sleep
    return out


def archive(run: Path) -> tuple[dict, dict]:
    lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    recs = {}
    for d in lines[1:]:
        recs[d["key"]] = d
    return lines[0], recs


def test_search(train_file: Path, splits: Path, seeds: Path) -> list[str]:
    print("global search")
    run, cache = W / "run", W / "rounds.jsonl"
    common = ["--splits", str(splits), "--dataset", str(train_file), "--seeds", str(seeds), "--out", str(run),
              "--live-cache", str(cache), "--workers", "8"] + V4_EXEC
    run_cli(V4, common + ["--generations", "2"])
    header, recs = archive(run)
    split = SPL.load_split(splits)
    check(header["version"] == 4 and header["qids"] == split["train"] and header["dev"] == split["dev"],
          "the header names format version 4, the train questions and the dev questions")
    check(header["settings"].get("last_round_vote") is True and header["levels"] == [["cheap", 4.5], ["medium", 10.5],
                                                                                     ["expensive", None]],
          "the header records the repair and the levels")
    seeds_ = [d for d in recs.values() if d["gen"] == 0]
    check(len(seeds_) == 11 and all(set(d["reps"]["0"]) == set(split["train"]) for d in seeds_),
          "every seed is scored once on every train question")
    check(all(set(d["reps"]) == {"0"} for d in recs.values()), "no program has a second run")
    check(all(not set(v) & set(split["dev"]) for d in recs.values() for v in d["reps"].values()),
          "no dev question is in the search archive")
    gens = [json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
    check([g["gen"] for g in gens] == [0, 1, 2], "generations 0, 1, 2 logged")
    check(all(g["parents"] <= 3 and g["children"] <= 5 * g["parents"] for g in gens[1:]),
          "at most 3 parents and 5 children each", str([(g["parents"], g["children"]) for g in gens[1:]]))
    kids = [d for d in recs.values() if d["gen"] > 0]
    check(kids and all(d["meta"]["slot"] in ("cheap", "medium", "expensive") and d["meta"]["target"] == "all"
                       for d in kids), "every child records its parent's level")
    kinds = P.edit_kinds()                  # as the search CLI left them: set_width for plan_width
    check(P.WIDTH_OP in kinds and all(d["op"] in kinds for d in kids),
          "every child is one named edit of this run's kinds (with set_width)", str(Counter(d["op"] for d in kids)))
    check(P.MAX_TURNS == 15 and D.HIGH_COST == 3
          and all(P.plan_cost(d["program"]) <= P.MAX_TURNS for d in recs.values()),
          "no program's plan rounds cost more than the 15-turn cap (high effort = 3 turns)")
    # the slots, recomputed from the archive on file
    s = V4.Search(types.SimpleNamespace(out=str(W / "check"), seed=0, max_calls_per_question=15, splits=splits),
                  {}, split, None, {})
    for d in recs.values():
        r = P.ProgRecord.from_json(d)
        r.dup_of = None
        s.archive[r.key] = r
    for r in s.archive.values():
        s.mark_duplicate(r)
    slots = s.compute_slots()
    logged = {(x["level"], "best"): x["key"] for x in gens[-1]["slots"]}
    check(slots == logged, "the logged slots follow from the archive")
    pool = s.eligible()
    ok = True
    for (level, _), key in slots.items():
        members = [r for r in pool if V4.level_of(r.turns(s.qids)) == level]
        ok &= s.archive[key] in members
        ok &= abs(s.archive[key].score(s.qids) - max(r.score(s.qids) for r in members)) < 1e-9
    check(ok and len(slots) >= 2, "each slot holds the best program of its level", str(len(slots)))
    summary = json.loads((run / "summary.json").read_text())
    check(summary["version"] == 4 and {x["level"] for x in summary["slots"]} == {l for l, _ in slots},
          "summary.json lists the level holders")
    # resume
    before = T.MODEL_CALLS["chat"]
    run_cli(V4, common + ["--generations", "3", "--resume"])
    gens = [json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
    check(gens[-1]["gen"] == 3 and T.MODEL_CALLS["chat"] > before, "resume runs generation 3")
    before = T.MODEL_CALLS["chat"]
    run_cli(V4, common + ["--generations", "3", "--resume"])
    check(T.MODEL_CALLS["chat"] == before, "a finished search resumes without spending")
    try:
        run_cli(V4, common + ["--generations", "3", "--resume", "--seed", "5"])
        check(False, "a different --seed is refused on resume")
    except SystemExit:
        check(True, "a different --seed is refused on resume")
    other = W / "other_split.json"
    other.write_text(json.dumps({"train": split["train"][:-1], "dev": split["dev"] + split["train"][-1:]}))
    try:
        run_cli(V4, [a if a != str(splits) else str(other) for a in common] + ["--generations", "3", "--resume"])
        check(False, "a different split is refused on resume")
    except SystemExit:
        check(True, "a different split is refused on resume")
    # the dev guard
    try:
        s.run_jobs([(next(iter(s.archive.values())), split["dev"][0], 0)], "x")
        check(False, "the search refuses to run a dev question")
    except RuntimeError:
        check(True, "the search refuses to run a dev question")
    # the final choice on the dev split
    n_lines = sum(1 for _ in (run / "archive.jsonl").open())
    run_cli(V4, common + ["--select-dev"])
    final = json.loads((run / "final.json").read_text())
    header, recs = archive(run)
    check(sum(1 for _ in (run / "archive.jsonl").open()) == n_lines
          and all(not set(v) & set(split["dev"]) for d in recs.values() for v in d["reps"].values()),
          "the dev results never enter the archive")
    s2 = V4.Search(types.SimpleNamespace(out=str(W / "check2"), seed=0, max_calls_per_question=15, splits=splits),
                   {}, split, None, {})
    for d in recs.values():
        r = P.ProgRecord.from_json(d)
        r.dup_of = None
        s2.archive[r.key] = r
    for r in s2.archive.values():
        s2.mark_duplicate(r)
    want = s2.dev_candidates()
    ok = True
    for level, d in final["levels"].items():
        cands = d["candidates"]
        ok &= [c["key"] for c in cands] == [r.key for r in want[level]]
        ok &= len(cands) <= 3 and all(set(c["dev_marks"]) == set(split["dev"]) for c in cands)
        if cands:
            best = max(c["dev_acc"] for c in cands)
            chosen = next(c for c in cands if c["key"] == d["chosen"])
            ok &= abs(chosen["dev_acc"] - best) < 1e-9
            ok &= abs(chosen["dev_acc"] - sum(chosen["dev_marks"].values()) / len(split["dev"])) < 1e-9
            ok &= sum(c["seed"] for c in cands) >= 1 or not any(r.gen == 0 for r in s2.eligible()
                                                                  if s2.level(r) == level)
    check(ok, "per level: the 2 best on train and the best seed, the chosen one best on dev")
    check((run / "final.md").exists(), "final.md written")
    before = T.MODEL_CALLS["chat"]
    run_cli(V4, common + ["--select-dev"])
    check(T.MODEL_CALLS["chat"] == before and json.loads((run / "final.json").read_text()) == final,
          "the final choice redone costs nothing and gives the same result")
    return split["train"] + split["dev"]


class _FrozenSalt:
    """Stands in for the fake model's call counter: every answer then depends only on the
    conversation, not on the order threads call in."""

    def __iadd__(self, other):
        return self

    def __str__(self):
        return "0"

    __format__ = lambda self, spec: "0"


def test_redo(train_file: Path, splits: Path, seeds: Path):
    print("interrupted generations")
    def common(run, cache):
        return ["--splits", str(splits), "--dataset", str(train_file), "--seeds", str(seeds), "--out", str(run),
                "--live-cache", str(cache), "--workers", "8"] + V4_EXEC

    def gen2(run):
        _, recs = archive(run)
        return {k: (d["name"], d["parent"], d["op"], d["meta"]["slot"], json.dumps(d["reps"], sort_keys=True))
                for k, d in recs.items() if d["gen"] == 2}

    old_salt = T.FakeCompletions.salt
    T.FakeCompletions.salt = _FrozenSalt()
    orig = V4.Search.run_jobs
    try:
        # A: uninterrupted, from an empty cache
        a_run, a_cache = W / "redoA", W / "redoA.jsonl"
        run_cli(V4, common(a_run, a_cache) + ["--generations", "2"])
        ga = [json.loads(l) for l in (a_run / "generations.jsonl").open() if l.strip()]
        for when, label in (("before", "children drawn and on file, not run"),
                            ("after", "children run and recorded in the cache, not yet saved with results")):
            # B: its own empty cache, stopped in generation 2 wave 3, then resumed
            b_run, b_cache = W / f"redoB_{when}", W / f"redoB_{when}.jsonl"

            def boom(self, jobs, lab, when=when):
                if lab.startswith("gen 2 wave 3"):
                    if when == "after":
                        orig(self, jobs, lab)
                    raise KeyboardInterrupt
                return orig(self, jobs, lab)
            V4.Search.run_jobs = boom
            try:
                run_cli(V4, common(b_run, b_cache) + ["--generations", "2"])
            finally:
                V4.Search.run_jobs = orig
            gens = [json.loads(l)["gen"] for l in (b_run / "generations.jsonl").open() if l.strip()]
            on_file = gen2(b_run)
            check(gens == [0, 1] and len(on_file) > 0,
                  f"a run stopped in generation 2 wave 3 ({label}) keeps its children on file and logs 0 and 1")
            run_cli(V4, common(b_run, b_cache) + ["--generations", "2", "--resume"])
            gb = [json.loads(l) for l in (b_run / "generations.jsonl").open() if l.strip()]
            check(gen2(a_run) == gen2(b_run), f"the redone generation ({label}) equals the uninterrupted one: "
                  "children, names, parents, edits and results", f"{len(gen2(a_run))} vs {len(gen2(b_run))}")
            check(ga[-1]["slots"] == gb[-1]["slots"] and ga[-1]["children"] == gb[-1]["children"]
                  and gb[-1].get("redone_child", 0) == len(on_file),
                  f"the same slots and child count; every child on file is reused ({label})",
                  f"{gb[-1].get('redone_child')} of {len(on_file)}")
    finally:
        V4.Search.run_jobs = orig
        T.FakeCompletions.salt = old_salt
    b_run = W / "redoB_after"
    summary = json.loads((b_run / "summary.json").read_text())
    check(summary["generations"] == 2, "summary.json counts the logged generations")

    # a short program of an earlier generation is completed when the next generation starts
    d_run, d_cache = W / "redoD", W / "redoD.jsonl"
    shutil.copytree(a_run, d_run)
    shutil.copy(a_cache, d_cache)
    _, rd = archive(d_run)
    victim = next(d for d in rd.values() if d["gen"] == 1)
    cut = dict(victim)
    cut["reps"] = {"0": dict(list(victim["reps"]["0"].items())[:5])}
    with (d_run / "archive.jsonl").open("a") as fh:
        fh.write(json.dumps(cut) + "\n")
    run_cli(V4, common(d_run, d_cache) + ["--generations", "3", "--resume"])
    _, rd = archive(d_run)
    gd = [json.loads(l) for l in (d_run / "generations.jsonl").open() if l.strip()]
    check(len(rd[victim["key"]]["reps"]["0"]) == len(victim["reps"]["0"]) and gd[-1].get("completed_short") == 1,
          "a program left short is completed at the next generation")

    # the generation-0 line lost after the seeds were saved is written on resume
    c_run, c_cache = W / "redoC", W / "redoC.jsonl"
    shutil.copy(a_cache, c_cache)
    run_cli(V4, common(c_run, c_cache) + ["--generations", "0"])
    (c_run / "generations.jsonl").write_text("")
    run_cli(V4, common(c_run, c_cache) + ["--generations", "0", "--resume"])
    gc = [json.loads(l) for l in (c_run / "generations.jsonl").open() if l.strip()]
    check([g["gen"] for g in gc] == [0] and gc[0].get("logged_on_resume"),
          "a lost generation-0 line is written again on resume")


class _Down:
    """The model server is down: every request fails."""

    def __enter__(self):
        self.old = T.FakeCompletions.create

        def create(*a, **k):
            raise ConnectionError("Connection refused")
        T.FakeCompletions.create = create
        return self

    def __exit__(self, *exc):
        T.FakeCompletions.create = self.old


def test_failures(train_file: Path, splits: Path, seeds: Path):
    print("server outages, torn lines, mismatched files")
    a_run, a_cache = W / "redoA", W / "redoA.jsonl"
    def common(run, cache, seeds_=seeds):
        return ["--splits", str(splits), "--dataset", str(train_file), "--seeds", str(seeds_), "--out", str(run),
                "--live-cache", str(cache), "--workers", "8"] + V4_EXEC
    # an outage in generation 3: the search stops before logging it, and a rerun redoes it
    o_run, o_cache = W / "outage", W / "outage.jsonl"
    shutil.copytree(a_run, o_run)
    shutil.copy(a_cache, o_cache)
    with _Down():
        try:
            run_cli(V4, common(o_run, o_cache) + ["--generations", "3", "--resume"])
            check(False, "an outage stops the search")
        except SystemExit as exc:
            gens = [json.loads(l)["gen"] for l in (o_run / "generations.jsonl").open() if l.strip()]
            check(bool(exc.code) and "could not be completed" in str(exc.code) and gens == [0, 1, 2],
                  "an outage stops the search with a message, the generation unlogged", str(exc.code)[:200])
    run_cli(V4, common(o_run, o_cache) + ["--generations", "3", "--resume"])
    _, ro = archive(o_run)
    go = [json.loads(l) for l in (o_run / "generations.jsonl").open() if l.strip()]
    check([g["gen"] for g in go] == [0, 1, 2, 3] and go[-1].get("redone_child", 0) > 0
          and all(len(d["reps"].get("0", {})) == len(SPL.load_split(splits)["train"]) for d in ro.values()),
          "after the outage a rerun redoes the generation; every program has every train result")
    # an outage in generation 0: no generation is logged, and a rerun completes the seeds in place
    z_run, z_cache = W / "outage0", W / "outage0.jsonl"
    with _Down():
        try:
            run_cli(V4, common(z_run, z_cache) + ["--generations", "0"])
            check(False, "an outage in generation 0 stops the search")
        except SystemExit as exc:
            check(bool(exc.code) and not [l for l in (z_run / "generations.jsonl").open() if l.strip()]
                  if (z_run / "generations.jsonl").exists() else bool(exc.code),
                  "an outage in generation 0 stops the search before generation 0 is logged")
    run_cli(V4, common(z_run, z_cache) + ["--generations", "0", "--resume"])
    _, rz = archive(z_run)
    gz = [json.loads(l)["gen"] for l in (z_run / "generations.jsonl").open() if l.strip()]
    n_train = len(SPL.load_split(splits)["train"])
    check(gz == [0] and len(rz) == len(json.loads(seeds.read_text())["seeds"])
          and all(len(d["reps"].get("0", {})) == n_train for d in rz.values()),
          "after the outage the rerun completes every seed on every train question and logs generation 0")
    # torn last lines (a hard stop mid-write) are skipped, and the next record starts a new line
    t_run, t_cache = W / "torn", W / "torn.jsonl"
    shutil.copytree(a_run, t_run)
    shutil.copy(a_cache, t_cache)
    with (t_run / "archive.jsonl").open("a") as fh:
        fh.write('{"key": "torn", "program": {"pl')
    with (t_run / "generations.jsonl").open("a") as fh:
        fh.write('{"gen": 3, "chil')
    run_cli(V4, common(t_run, t_cache) + ["--generations", "3", "--resume"])
    bad = 0
    for line in (t_run / "archive.jsonl").open():
        try:
            json.loads(line)
        except ValueError:
            bad += 1
    gt = [json.loads(l) for l in (t_run / "generations.jsonl").read_text().splitlines()
          if l.strip() and l != '{"gen": 3, "chil']
    check(bad == 1 and [g["gen"] for g in gt] == [0, 1, 2, 3],
          "a torn last line is skipped; the records after it are read", f"{bad} bad lines")
    # a seed file written under other settings, or from other questions, is refused
    sd = json.loads(seeds.read_text())
    for label, other in (("other settings", {**sd, "settings": {**sd["settings"], "turn_cap": 16}}),
                         ("examples outside the train split", {**sd, "examples": SPL.load_split(splits)["dev"][:2]})):
        path = W / f"seeds_other_{len(label)}.json"
        path.write_text(json.dumps(other))
        try:
            run_cli(V4, common(W / f"fresh_{len(label)}", W / "fresh.jsonl", path) + ["--generations", "0"])
            check(False, f"a seed file with {label} is refused")
        except SystemExit as exc:
            check(bool(exc.code), f"a seed file with {label} is refused")
    # an interrupted final choice exits with an error and writes no final.json
    i_run = W / "interrupted_dev"
    shutil.copytree(a_run, i_run)
    (i_run / "final.json").unlink(missing_ok=True)
    old = V4.Search.select_on_dev
    V4.Search.select_on_dev = lambda self: (_ for _ in ()).throw(KeyboardInterrupt())
    try:
        run_cli(V4, common(i_run, a_cache) + ["--select-dev"])
        check(False, "an interrupted final choice exits with an error")
    except SystemExit as exc:
        check(bool(exc.code) and not (i_run / "final.json").exists(),
              "an interrupted final choice exits with an error and writes no final.json")
    finally:
        V4.Search.select_on_dev = old


def test_eval(test_file: Path, test_ids: list[str], used: list[str]):
    print("v4 test evaluation")
    run = W / "run"
    ext = W / "external_sr.json"
    rows = {r["id"]: r for r in json.loads(test_file.read_text())}
    def wrong(q):
        return next(l for l in "ABCD" if l != rows[q]["answer_letter"])
    ext.write_text(json.dumps({"per_question": {q: {"answer": rows[q]["answer_letter"],
                                                    "preds": [rows[q]["answer_letter"], wrong(q), wrong(q)],
                                                    "tokens": [100, 120, 140]} for q in test_ids}}))
    argv = ["--run", str(run), "--dataset", str(test_file), "--reps", "2", "--workers", "8",
            "--external", f"self-refine={ext}"] + V4_EXEC
    run_cli(E4, argv)
    res = json.loads((run / "test_eval" / "results_k2.json").read_text())
    finals = [f"v4 {l}" for l, _ in V4.LEVELS if json.loads((run / "final.json").read_text())["levels"][l]["chosen"]]
    check(set(finals) <= set(res["table"]) and {"in-executor direct_high", "in-executor self_refine_high",
                                                 "external self-refine"} <= set(res["table"]),
          "rows: the final programs, the in-executor baselines and the external one", str(list(res["table"])))
    check(abs(res["table"]["external self-refine"]["avg@2"] - 0.5) < 1e-9, "external rows read their first 2 runs")
    check(all(f"{f} - external self-refine" in res["differences"] for f in finals),
          "every final program is compared with self-refine")
    check(all(abs(sum(r["share"] for r in res["paths"][f]) - 1) < 1e-9
              and sum(r["debates"] for r in res["paths"][f]) == 2 * len(test_ids) for f in finals),
          "each program's paths cover all its test debates")
    check(all(abs(sum(r["share"] for r in res["stops"][f]) - 1) < 1e-9 and res["stops"][f] for f in finals),
          "each program's stopping rules cover all its test debates", str({f: [r["stop"] for r in res["stops"][f]]
                                                                           for f in finals}))
    svg = (run / "test_eval" / "accuracy_vs_tokens.svg").read_text()
    root = ET.fromstring(svg)
    dots = {res["roles"].get(name, name) for name, d in res["table"].items() if d.get("tokens") is not None}
    check(root.tag.endswith("svg") and svg.count("<title>") == len(dots),
          "the graph is valid SVG with one dot per program that has tokens", f"{svg.count('<title>')} vs {len(dots)}")
    check(E4.stop_label({"rules": [P.CONT(), {"when": ["step==1"], "do": "stop:vote"}], "default": "stop:last_commit"},
                        "0,1") == "rule 1: if step==1, stop:vote"
          and E4.stop_label({"rules": [P.CONT()], "default": "stop:last_commit"}, "0,0").endswith(
              "(plan used up, turn cap or step limit: stop:last_commit)")
          and E4.stop_label({"rules": [P.CONT()], "default": "stop:vote"}, "0,-1") == "no rule held: stop:vote",
          "the stopping rule is read off the recorded decisions")
    one = [{"label": "a", "x": 0, "y": 40.0, "se": 0.0, "kind": "v4"}]
    check(ET.fromstring(E4.scatter_svg(one, "t")) is not None, "the graph survives zero tokens and zero spread")
    before = T.MODEL_CALLS["chat"]
    run_cli(E4, argv)
    check(T.MODEL_CALLS["chat"] == before, "a rerun costs nothing")
    # debates that cannot be run: the tables say so and the run ends with an error
    with _Down():
        try:
            run_cli(E4, argv + ["--out", str(W / "eval_down"), "--live-cache", str(W / "eval_down.jsonl")])
            check(False, "missing test debates end the evaluation with an error")
        except SystemExit as exc:
            md = (W / "eval_down" / "results_k2.md").read_text()
            check(bool(exc.code) and "Incomplete:" in md,
                  "missing test debates are flagged in the tables and end the evaluation with an error")
    try:
        run_cli(E4, [a if not a.startswith("self-refine=") else "self-refine" for a in argv])
        check(False, "an --external pair without '=' is refused")
    except SystemExit as exc:
        check("label=path" in str(exc.code), "an --external pair without '=' is refused with a message")
    mixed = W / "mixed_test.json"
    train_rows = json.loads((W / "toy_train.json").read_text())
    mixed.write_text(json.dumps(list(rows.values()) + [r for r in train_rows if r["id"] == used[0]]))
    try:
        run_cli(E4, [a if a != str(test_file) else str(mixed) for a in argv])
        check(False, "a test set holding a train question is refused")
    except SystemExit:
        check(True, "a test set holding a train question is refused")


if __name__ == "__main__":
    W.mkdir(parents=True, exist_ok=True)
    test_read_repair()
    test_settings()
    test_level_labels()
    test_seed_writer()
    test_levels_and_splits()
    train_file, test_file, splits, test_ids = toy_data()
    seeds = test_seeds(train_file, splits)
    used = test_search(train_file, splits, seeds)
    test_redo(train_file, splits, seeds)
    test_failures(train_file, splits, seeds)
    test_eval(test_file, test_ids, used)
    print()
    if T.FAILURES:
        print(f"{len(T.FAILURES)} FAILED: " + "; ".join(T.FAILURES))
        sys.exit(1)
    print(f"all checks passed (model calls: {T.MODEL_CALLS}); temp dir {TMP}")

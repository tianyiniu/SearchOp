"""Offline checks for evolve_program_mcq's live mode. The model is replaced by
a stub that answers deterministically, so the budget, prescreen, cache
write-and-reuse, per-question call cap, fresh-replicate recheck and --save-all
paths all run end to end with no vLLM and no cost.

    python3 scripts/test_evolve_program_live.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402

FAILURES: list[str] = []
CALLS = {"n": 0}


def check(okay: bool, label: str, extra: str = "") -> None:
    print(("  ok   " if okay else "  FAIL ") + label + (f"  [{extra}]" if extra else ""))
    if not okay:
        FAILURES.append(label)


def stub_execute_round(client, model, question, options, round_spec, all_rounds,
                       temperature, max_tokens=3072, prompts=None):
    """One canned response per persona: a letter chosen from a hash of the
    question and the position, so reruns agree and replicates can differ."""
    out = []
    for i, persona in enumerate(round_spec["personas"]):
        CALLS["n"] += 1
        seed = f"{question[:40]}|{len(all_rounds)}|{i}|{persona}"
        letter = "ABCD"[int(hashlib.sha1(seed.encode()).hexdigest(), 16) % 4]
        if persona in D.NON_ANSWERING:
            out.append((persona, "Ruling out options. SURVIVING: A B"))
        else:
            out.append((persona, f"Reasoning as {persona}.\nANSWER: {letter}"))
    return out


D.execute_round = stub_execute_round

rows_all = json.loads(Path("datasets/supergpqa_strict_train.json").read_text())
ROWS = {r["id"]: r for r in rows_all[:24]}
TMP = Path(tempfile.mkdtemp(prefix="evolve_live_test_"))
EMPTY = TMP / "empty_cache.jsonl"
EMPTY.write_text("")


def make_runner(live_cache: Path) -> M.BudgetedRunner:
    return M.BudgetedRunner(ROWS, [EMPTY], base_urls="http://127.0.0.1:1/v1",
                            model="stub", temperature=0.7, answer_tokens=64,
                            cache_path=live_cache, max_calls=10_000,
                            api_key="EMPTY", progress=False)


def test_budget_and_cache():
    print("budget, cache write, cache reuse")
    runner = make_runner(TMP / "live1.jsonl")
    row = next(iter(ROWS.values()))
    prog = M.chain("critic")
    CALLS["n"] = 0
    runner.reset_budget(0)
    try:
        M.run_program(prog, runner, row)
        check(False, "budget 0 raises OffCache on a miss")
    except M.OffCache:
        check(CALLS["n"] == 0, "budget 0 makes no model calls")
    runner.reset_budget(10)
    out = M.run_program(prog, runner, row)
    check(out["n_calls"] == 2 and CALLS["n"] == 2, "budget 10 runs solver+critic live",
          str(CALLS["n"]))
    runner.reset_budget(0)
    out2 = M.run_program(prog, runner, row)
    check(out2["letter"] == out["letter"] and CALLS["n"] == 2,
          "same request again is a cache hit, no new calls")
    runner.reset_budget(1)
    try:
        M.run_program(M.chain("critic", "fresh"), runner, row)   # fresh costs 2
        check(False, "budget 1 cannot afford a 2-call round")
    except M.OffCache:
        check(CALLS["n"] == 2, "budget too small -> OffCache before any call, prefix reused")
    runner.close()
    lines = [json.loads(l) for l in (TMP / "live1.jsonl").open() if l.strip()]
    check(len(lines) == 2, "two new rounds written to the live cache file", str(len(lines)))
    runner2 = make_runner(TMP / "live1.jsonl")            # resume
    runner2.reset_budget(0)
    CALLS["n"] = 0
    M.run_program(prog, runner2, row)
    check(CALLS["n"] == 0, "a new runner reloads the live cache and replays it")
    check((TMP / "live1.lock").exists(), "lock file exists while the cache is open")
    try:
        make_runner(TMP / "live1.jsonl")
        check(False, "second runner on an open cache is refused")
    except SystemExit as exc:
        check("in use by another run" in str(exc), "second runner on an open cache is refused")
    runner2.close()
    check(not (TMP / "live1.lock").exists(), "lock file removed on close")
    import socket
    (TMP / "live1.lock").write_text(json.dumps({"host": socket.gethostname(), "pid": 999999999,
                                                "started": "earlier"}))
    runner3 = make_runner(TMP / "live1.jsonl")
    check(runner3._own_lock, "stale lock from a dead pid on this host is cleared")
    runner3.close()
    (TMP / "live1.lock").write_text(json.dumps({"host": "some-other-host", "pid": 1,
                                                "started": "earlier"}))
    try:
        make_runner(TMP / "live1.jsonl")
        check(False, "lock from another host is respected")
    except SystemExit:
        check(True, "lock from another host is respected")
    (TMP / "live1.lock").unlink()


def test_torn_records_are_skipped():
    print("torn cache lines that still parse as JSON are skipped")
    torn = TMP / "torn.jsonl"
    good = {"q": "x", "k": json.dumps({"rounds": [{"personas": ["solver"]}]}), "r": 0,
            "responses": [["solver", "ANSWER: A"]], "error": None}
    bad1 = dict(good, q="y", responses=[["solver"]])             # lost the text
    bad2 = dict(good, k='{"rounds":[{"persexpert"]')             # spliced key
    bad3 = dict(good, q="z", responses="ANSWER: A")              # not a list
    torn.write_text("\n".join(json.dumps(x) for x in (bad1, good, bad2, bad3)) + "\n")
    target = {}
    M.load_cache_into(target, torn)
    check(len(target) == 1 and ("x", good["k"], 0) in target, "only the well-formed record loaded",
          str(len(target)))
    runner = M.BudgetedRunner({}, [], base_urls="http://127.0.0.1:1/v1", model="stub",
                              temperature=0.7, answer_tokens=64, cache_path=torn,
                              max_calls=10, api_key="EMPTY", progress=False)
    check(len(runner._cache) == 1, "live runner drops malformed rounds from its own file",
          str(len(runner._cache)))
    runner.close()


def test_digest_window_and_cache_key():
    print("digest window: head+tail clipping and cache-key separation")
    import schema_fitness as SF
    text = "A" * 100 + "B" * 2000 + "C" * 100 + "\nANSWER: D"
    try:
        check(D.digest_signature() is None, "default window is not part of the key")
        d0 = D._clip(text)
        check(d0 == text.strip()[:700] and "ANSWER" not in d0,
              "default: head-only cut, conclusion lost", f"{len(d0)} chars")
        spec = [{"personas": ["critic"], "sees": "all"}]
        k_default = SF.path_key(spec)
        D.set_digest(300, 900)
        d1 = D._clip(text)
        check(d1.startswith("A" * 100) and d1.rstrip().endswith("ANSWER: D"),
              "head+tail: keeps the opening AND the conclusion", f"{len(d1)} chars")
        check("omitted" in d1, "elision marker names what was dropped")
        check(D.digest_signature() == (300, 900), "non-default window is reported")
        check(SF.path_key(spec) != k_default,
              "cache key changes with the window, so recordings cannot mix")
        short = "short answer\nANSWER: B"
        check(D._clip(short) == short, "a response inside the budget is untouched")
    finally:
        D.set_digest(700, 0)
    check(D.digest_signature() is None and SF.path_key([{"personas": ["critic"], "sees": "all"}]) == k_default,
          "restoring the default restores the original key")


def test_no_eliminator():
    print("--no-eliminator removes the moves and the seeds that need them")
    import importlib
    m = importlib.reload(M)
    check("eliminate" in m.ACTIONS and "elim_blind" in m.ACTIONS, "eliminator moves present by default")
    check(any(r["do"] == "elim_blind" for r in m.PROGRAM_B["rules"]),
          "experiment B's parked track uses the eliminator")
    m.drop_eliminator()
    check("eliminate" not in m.ACTIONS and "elim_blind" not in m.ACTIONS, "moves dropped")
    check("eliminator" not in m.ROUND_KINDS, "eliminator gone from the round-kind menu",
          str(m.ROUND_KINDS))
    try:
        m.validate_program(m.PROGRAM_B)
        check(False, "the old program_b should no longer validate")
    except AssertionError:
        check(True, "the old program_b no longer validates")
    m.validate_program(m.PROGRAM_B_NOELIM)
    check(True, "the eliminator-free seed validates")
    rng = random.Random(3)
    prog = m.PROGRAM_B_NOELIM
    for _ in range(300):
        prog = m.mutate_program(prog, rng)
        conds = [c for r in prog["rules"] for c in r["when"]]
        acts = [r["do"] for r in prog["rules"]]
        if any("elim" in c for c in conds) or any("elim" in a for a in acts):
            check(False, "a mutation reintroduced an eliminator move")
            break
    else:
        check(True, "300 mutations never reintroduce the eliminator")
    importlib.reload(M)          # leave the module as the other tests expect


def test_call_cap_and_round_kinds():
    print("per-question call cap and round-content conditions")
    runner = make_runner(TMP / "live2.jsonl")
    runner.reset_budget(None)
    row = next(iter(ROWS.values()))
    out = M.run_program(M.chain("critic", "critic", "critic"), runner, row, max_calls=3)
    check(out["n_calls"] == 3, "cap 3 stops after solver + 2 critics", str(out["n_calls"]))
    prog = {"plan": [M.TG_ROOT],
            "rules": [{"when": ["acts==0"], "do": "continue"},
                      {"when": ["last_round:solver"], "do": "critic"},
                      {"when": ["last_round:critic", "ran_round:solver"], "do": "verifier"}],
            "default": "stop:last_commit"}
    out = M.run_program(prog, runner, row)
    check(out["actions"] == ["continue", "critic", "verifier"],
          "last_round:/ran_round: see plan rounds by content", str(out["actions"]))
    check("solver" in M.ROUND_KINDS and "expert+solver" in M.ROUND_KINDS,
          "round kind menu built from the action specs")
    runner.close()
    for bad, label in (({"plan": [M.TG_ROOT], "rules": [{"when": ["stepp==1"], "do": "critic"}],
                         "default": "stop:last_commit"}, "unknown condition rejected"),
                       ({"plan": [M.TG_ROOT], "rules": [], "default": "critic"},
                        "non-stop default rejected"),
                       ({"plan": [M.TG_ROOT], "rules": [{"when": ["last_round:nobody"],
                                                          "do": "critic"}],
                         "default": "stop:last_commit"}, "unknown round kind rejected")):
        try:
            M.validate_program(bad)
            check(False, label)
        except (AssertionError, ValueError):
            check(True, label)
    for name, prog in {"program_b": M.PROGRAM_B, "program_fixed": M.PROGRAM_FIXED,
                       **M.PROBES}.items():
        M.validate_program(prog)
    check(True, "all seed programs still validate")


def test_eval_merge():
    print("eval_program: subset fill merges into an existing Eval")
    runner = make_runner(TMP / "live3.jsonl")
    qids = list(ROWS)[:6]
    prog = M.chain("critic")
    runner.reset_budget(0)
    ev = M.eval_program(prog, runner, ROWS, qids, 0)
    check(ev.covered == 0 and ev.marks.count("x") == 6, "all off-cache on an empty cache")
    runner.reset_budget(None)
    gaps = [i for i, m in enumerate(ev.marks) if m == "x"][:4]
    M.eval_program(prog, runner, ROWS, qids, 0, workers=2, only=gaps, into=ev)
    check(ev.covered == 4 and ev.marks[4] == "x", "only the requested gaps were filled",
          "".join(ev.marks))
    check(len(ev.paths) == 1 and ev.path_ids[0] == 0, "action paths recorded and indexed")
    check(ev.letters[0] != "?" and ev.calls[0] == 2, "letters and calls recorded")
    j = ev.to_json("train")
    check(set(j) >= {"train_outcomes", "train_letters", "train_calls", "train_path_ids",
                     "train_paths"}, "to_json carries the clustering fields")
    runner.close()


def test_evolve_live_end_to_end():
    print("evolve --live end to end on 24 questions with the stub model")
    ds = TMP / "mini.json"
    ds.write_text(json.dumps(list(ROWS.values())))
    args = argparse.Namespace(
        seed=0, dataset=ds, cache=EMPTY, treegrow_cache=EMPTY, rep=0, limit=None,
        max_calls_per_question=8, live=True, live_cache=TMP / "live4.jsonl",
        base_urls="http://127.0.0.1:1/v1", model="stub", temperature=0.7,
        answer_tokens=64, api_key="EMPTY", novelty_budget=40, prescreen_margin=1.0,
        min_coverage=0.0, workers=4, recheck_top=2, recheck_rep=1, recheck_n=8,
        recheck_baselines=True, ignore_cache_lock=False, no_eliminator=False,
        max_total_calls=100_000, generations=2, population=4, offspring=2,
        evolved_out=TMP / "evolved.json", save_all=TMP / "all.jsonl",
        bestofn_cache=Path("does/not/exist.jsonl"))
    CALLS["n"] = 0
    M.evolve(args)
    out = json.loads((TMP / "evolved.json").read_text())
    check(out["live"] is True and out["live_calls_total"] > 0, "live calls were made",
          str(out["live_calls_total"]))
    top = out["top"][0]
    check(top["dev_covered"] == 1.0, "dev gaps were filled live, not scored as wrong",
          str(top["dev_covered"]))
    check("dev_fresh_acc" in top and top["dev_fresh_rep"] == 1,
          "top programs got a fresh-replicate dev recheck")
    check(top["dev_fresh_n"] == 8 and len(out["recheck_qids"]) == 8
          and len(top["dev_fresh_outcomes"]) == 8, "recheck ran on the fixed 8-question subset")
    check(set(out["recheck_qids"]) <= set(json.loads((TMP / "all.jsonl").open().readline())["dev_qids"]),
          "recheck subset drawn from dev only")
    check("dev_paths" in top and "dev_fresh_letters" in top, "dev details saved")
    lines = [json.loads(l) for l in (TMP / "all.jsonl").open() if l.strip()]
    header, recs = lines[0], lines[1:]
    check("questions" in header and len(header["questions"]) == 24,
          "save-all header has per-question features")
    feat = next(iter(header["questions"].values()))
    check({"discipline", "difficulty", "n_options", "gold", "r1_majority"} <= set(feat),
          "feature fields present", str(sorted(feat)))
    check(all("train_paths" in r and "live_calls_spent" in r for r in recs),
          "every scored program carries paths and live spend")
    check(any(r["live_calls_spent"] > 0 for r in recs), "some candidates spent budget")
    check(all(r["live_calls_spent"] <= 40 + 8 for r in recs),
          "no candidate exceeded budget + one question's cap",
          str(max(r["live_calls_spent"] for r in recs)))
    check(set(out["baselines_fresh"]) == {"program_b", "program_fixed", "solver_critic"},
          "reference programs got the same fresh dev run")


if __name__ == "__main__":
    test_budget_and_cache()
    test_torn_records_are_skipped()
    test_digest_window_and_cache_key()
    test_no_eliminator()
    test_call_cap_and_round_kinds()
    test_eval_merge()
    test_evolve_live_end_to_end()
    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
    sys.exit(1 if FAILURES else 0)

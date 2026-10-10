"""Offline test of route_questions.py and eval_routed_dev.py: fake debate model,
fake guide, a tiny search, and made-up dev vectors whose right group is known.

    python tests/test_routed_dev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))   # the code under test
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # the other test files

import test_pipeline_cluster as T3  # noqa: E402  (installs the fake clients on import)
import eval_routed_dev as EV  # noqa: E402
import evolve_pipeline_cluster as V  # noqa: E402
import program_space as P  # noqa: E402
import route_questions as RQ  # noqa: E402

T, check, run_cli, TMP = T3.T, T3.check, T3.run_cli, T3.TMP
DEV = P.ROOT / "tests/data/supergpqa_program_search_dev_small.json"
N_DEV = 36


def fake_dev_vectors() -> tuple[Path, dict[str, int]]:
    """Dev ids carrying the vectors of training questions, so the group each
    must be routed to is the stored group of the training question it copies."""
    train = np.load(P.ROOT / "tests/data/question_vectors_train.npz", allow_pickle=True)
    labels = json.loads(T3.CLUSTERS_V3.read_text())["labels"]
    dev_ids = [r["id"] for r in json.loads(DEV.read_text())][:N_DEV]
    pick = np.random.default_rng(0).choice(len(train["ids"]), N_DEV, replace=False)
    path = TMP / "vectors_dev.npz"
    np.savez_compressed(path, ids=np.array(dev_ids), multihot=train["multihot"][pick],
                        template=train["template"][pick], question=train["question"][pick],
                        multihot_names=train["multihot_names"], embed_model=train["embed_model"])
    return path, {d: labels[str(train["ids"][j])] for d, j in zip(dev_ids, pick)}


def test_routes() -> Path:
    print("routes")
    vectors, want = fake_dev_vectors()
    out = TMP / "routes.json"
    run_cli(RQ, ["--clusters", str(T3.CLUSTERS_V3), "--test-vectors", str(vectors), "--out", str(out)])
    d = json.loads(out.read_text())
    check(d["train_agreement"] == 1.0, "the rule returns every training question's stored group")
    check({q: r["group"] for q, r in d["routes"].items()} == want,
          "a dev question carrying a training question's vector gets that question's group")
    check(all(r["margin"] >= 0 and r["second"] != r["group"] and len(r["distances"]) == 6
              for r in d["routes"].values()), "margins and distances are recorded")
    check(sum(r["knn"] == r["group"] for r in d["routes"].values()) >= 0.8 * N_DEV,
          "the knn route mostly agrees")
    try:
        run_cli(RQ, ["--clusters", str(T3.CLUSTERS_V3), "--out", str(TMP / "bad.json"),
                     "--test-vectors", str(P.ROOT / "tests/data/question_vectors_train.npz")])
        check(False, "training questions are refused as test questions")
    except SystemExit:
        check(True, "training questions are refused as test questions")
    return out


def test_eval(seeds_path: Path, routes: Path):
    print("dev evaluation")
    run = TMP / "v3" / "run_dev"
    common = ["--out", str(run), "--clusters", str(T3.CLUSTERS_V3), "--per-group", "3",
              "--live-cache", str(T3.CACHE), "--workers", "8"] + T3.WINDOW
    run_cli(V, common + ["--seeds", str(seeds_path), "--generations", "1"])
    run_cli(V, common + ["--pick-champions", "--heldout-cap", "4"])
    before = dict(run_files(run))
    args = ["--run", str(run), "--routes", str(routes), "--workers", "8"] + T3.WINDOW
    run_cli(EV, args)
    check(dict(run_files(run)) == before, "the search run's own files are untouched")
    res = json.loads((run / "dev_eval" / "results_k3.json").read_text())
    first = json.loads((run / "dev_eval" / "results_k1.json").read_text())
    check(res["n_questions"] == N_DEV and not any(res["missing"].values()), "every debate was recorded")
    progs, role = res["programs"], res["roles"]
    check(set(role) == {f"{p}_{g}" for p in ("champion", "grid") for g in range(6)}
          | {"global", "direct", "mad", "self_refine"},
          "six champions, six grid programs, the global champion and three baselines")
    slots = {x["group"]: x["key"] for x in json.loads((run / "summary.json").read_text())["slots"]
             if x["slot"] == "A"}
    check(all(role[f"grid_{g}"] == slots[g] for g in range(6)), "the grid programs are the slot-A holders")
    r = json.loads(routes.read_text())["routes"]

    def marks(name, q):
        return [progs[role[name]]["reps"][rep][q][0] for rep in ("0", "1", "2")]

    for prefix, row in (("champion", "held-out champions"), ("grid", "strongest grid programs")):
        t = res["table"][f"routed, {row}"]
        picks = [marks(f"{prefix}_{r[q]['group']}", q) for q in r]
        check(abs(t["avg@3"] - sum(sum(m) / 3 for m in picks) / N_DEV) < 1e-9
              and abs(t["avg@1"] - sum(m[0] for m in picks) / N_DEV) < 1e-9
              and abs(t["pass@3"] - sum(max(m) for m in picks) / N_DEV) < 1e-9,
              f"routed, {row}: avg@1, avg@3 and pass@3 recomputed exactly", f"{t['avg@3']:.3f}")
        rand = sum(sum(marks(f"{prefix}_{g}", q)) / 3 for g in range(6) for q in r) / (6 * N_DEV)
        best = sum(max(sum(marks(f"{prefix}_{g}", q)) / 3 for g in range(6)) for q in r) / N_DEV
        check(abs(res["table"][f"random group, {row} (expected)"]["avg@3"] - rand) < 1e-9
              and abs(res["table"][f"best of {row} (bound)"]["avg@3"] - best) < 1e-9
              and best >= t["avg@3"] - 1e-9 and best >= rand - 1e-9, f"{row}: random-group and bound rows")
    check(all(d["avg@1"] <= d["pass@3"] + 1e-9 for d in res["table"].values()), "pass@3 is never below avg@1")
    check(all(res["table"][k]["tokens"] > 0 for k in res["table"] if not k.startswith("best of")),
          "every row reports tokens")
    check("| tokens |" in (run / "dev_eval" / "results_k3.md").read_text(), "the report has a tokens column")
    check(first["reps"] == 1 and all(set(p["reps"]) == {"0"} for p in first["programs"].values())
          and all(abs(first["table"][k]["avg@1"] - res["table"][k]["avg@1"]) < 1e-9 for k in res["table"]),
          "the first-replicate tables are written first and their avg@1 never changes")
    check(sum(g["n"] for g in res["per_group"].values()) == N_DEV, "the by-group table covers every question")
    calls = dict(T.MODEL_CALLS)
    run_cli(EV, args)
    check(dict(T.MODEL_CALLS) == calls, "a second run calls no model")
    check(json.loads((run / "dev_eval" / "results_k3.json").read_text())["table"] == res["table"],
          "and gives the same table")
    # without the champions file: grid programs only, the global slot's holder as the global one
    hidden = run / "champions.json.hidden"
    (run / "champions.json").rename(hidden)
    try:
        run_cli(EV, args + ["--no-champions", "--out", str(TMP / "grid_only"), "--live-cache",
                            str(run / "dev_eval" / next(f.name for f in (run / "dev_eval").glob("rounds_*.jsonl")))])
        g = json.loads((TMP / "grid_only" / "results_k3.json").read_text())
        top = next(x for x in json.loads((run / "summary.json").read_text())["slots"] if x["slot"] == "G")["program"]
        check(not any("champion" in k for k in g["table"]) and "routed, strongest grid programs" in g["table"]
              and "global slot holder" in g["table"],
              "--no-champions: grid rows only, with the global slot's holder as the global row")
        check(g["roles"]["global"] == P.canon(P.normalize_program(top))
              and abs(g["table"]["routed, strongest grid programs"]["avg@3"]
                      - res["table"]["routed, strongest grid programs"]["avg@3"]) < 1e-9,
              "--no-champions: same routed grid score as the full evaluation")
        # step 9 since 2026-10-09: no champions, no in-executor protocols (--baselines ""), and each
        # group's program on its own group's questions only (--routed-only)
        step9 = ["--no-champions", "--baselines", "", "--routed-only"]
        calls = dict(T.MODEL_CALLS)
        run_cli(EV, args + step9 + ["--out", str(TMP / "step9_cached"), "--live-cache",
                                    str(run / "dev_eval" / next(f.name for f in (run / "dev_eval")
                                                                .glob("rounds_*.jsonl")))])
        b = json.loads((TMP / "step9_cached" / "results_k3.json").read_text())
        check(set(b["roles"]) == {f"grid_{g}" for g in range(6)} | {"global"}
              and not any(k.startswith("in-executor") for k in b["table"])
              and b["table"]["routed, strongest grid programs"] == g["table"]["routed, strongest grid programs"]
              and b["table"]["global slot holder"] == g["table"]["global slot holder"]
              and dict(T.MODEL_CALLS) == calls,
              "step 9 on recorded debates: the grid programs and the global one only, the same scores, no model call",
              str(sorted(b["roles"])))
        check(not any(k.startswith(("random group", "best of")) or "knn route" in k or "question route" in k
                      for k in b["table"]) and not any("other_grid" in d for d in b["per_group"].values())
              and "unsure quarter -> global" in b["table"],
              "routed only: the rows that need other groups' programs are left out, the others kept")
        # from an empty cache: a group's program runs on its own group's questions only, the global one on
        # every question, and a program holding both roles runs each question once
        run_cli(EV, args + step9 + ["--out", str(TMP / "step9_fresh"), "--reps", "1"])
        f = json.loads((TMP / "step9_fresh" / "results_k1.json").read_text())
        own = {gg: {q for q in r if r[q]["group"] == gg} for gg in range(6)}
        ran = {k: set(p["reps"]["0"]) for k, p in f["programs"].items()}
        want: dict[str, set] = {}
        for name, k in f["roles"].items():
            want.setdefault(k, set()).update(set(r) if name == "global" else own[int(name.split("_")[1])])
        rounds = sum(1 for _ in (TMP / "step9_fresh").glob("rounds_*.jsonl"))
        check(ran == want and not any(f["missing"].values()) and rounds == 1,
              "routed only, from an empty cache: each program ran exactly the questions of its roles",
              f"{sum(map(len, ran.values()))} debates for {len(r)} questions")
        check(abs(f["table"]["routed, strongest grid programs"]["avg@1"]
                  - sum(f["programs"][f["roles"][f"grid_{r[q]['group']}"]]["reps"]["0"][q][0] for q in r) / N_DEV) < 1e-9,
              "routed only: the routed row is each question's own group program")
        bad = {d: md_table_errors(TMP / d / f"results_k{k}.md")
               for d, k in (("step9_cached", 3), ("step9_fresh", 1), ("grid_only", 3))}
        check(not any(bad.values()), "every report table has the same number of cells on every line", str(bad))
        # --pool-replicates (step 9 since 2026-10-09): every replicate in one pool, the tables at the end
        cache = str(run / "dev_eval" / next(f.name for f in (run / "dev_eval").glob("rounds_*.jsonl")))
        calls = dict(T.MODEL_CALLS)
        run_cli(EV, args + step9 + ["--pool-replicates", "--out", str(TMP / "step9_pooled_cached"), "--live-cache", cache])
        tables = {k: (json.loads((TMP / "step9_pooled_cached" / f"results_k{k}.json").read_text()),
                      json.loads((TMP / "step9_cached" / f"results_k{k}.json").read_text())) for k in (1, 2, 3)}
        check(all(a["table"] == b["table"] and a["reps"] == b["reps"] == k for k, (a, b) in tables.items())
              and dict(T.MODEL_CALLS) == calls,
              "pooled replicates on recorded debates: the same three tables as one replicate after another, no model call")
        run_cli(EV, args + step9 + ["--pool-replicates", "--out", str(TMP / "step9_pooled_fresh"), "--reps", "2"])
        pf = json.loads((TMP / "step9_pooled_fresh" / "results_k2.json").read_text())
        p1 = json.loads((TMP / "step9_pooled_fresh" / "results_k1.json").read_text())
        check(pf["roles"] == f["roles"]
              and all(set(p["reps"][rep]) == want[k] for k, p in pf["programs"].items() for rep in ("0", "1"))
              and not any(pf["missing"].values()) and p1["reps"] == 1
              and all(set(p["reps"]) == {"0"} for p in p1["programs"].values()),
              "pooled replicates from an empty cache: each replicate runs exactly the questions of each role, "
              "and results_k1 holds replicate 1 alone")
        calls = dict(T.MODEL_CALLS)
        run_cli(EV, args + step9 + ["--pool-replicates", "--out", str(TMP / "step9_pooled_fresh"), "--reps", "2"])
        check(dict(T.MODEL_CALLS) == calls
              and json.loads((TMP / "step9_pooled_fresh" / "results_k2.json").read_text())["table"] == pf["table"],
              "a pooled rerun calls no model and gives the same tables")
        try:                               # stopped part way (here by the call cap), then run again
            run_cli(EV, args + step9 + ["--pool-replicates", "--out", str(TMP / "step9_pooled_stop"), "--reps", "2",
                                        "--max-total-calls", "40"])
            check(False, "a pooled evaluation stopped part way stops nonzero")
        except SystemExit as exc:
            check("call cap stopped the evaluation" in str(exc)
                  and not (TMP / "step9_pooled_stop" / "results_k2.json").exists(),
                  "a pooled evaluation stopped part way stops nonzero, with no table", str(exc)[:80])
        run_cli(EV, args + step9 + ["--pool-replicates", "--out", str(TMP / "step9_pooled_stop"), "--reps", "2"])
        ps = json.loads((TMP / "step9_pooled_stop" / "results_k2.json").read_text())
        check(not any(ps["missing"].values())
              and all(set(p["reps"][rep]) == want[k] for k, p in ps["programs"].items() for rep in ("0", "1")),
              "run again, it finishes every debate of both replicates")
    finally:
        hidden.rename(run / "champions.json")
    # with the champions as well: each group's champion and grid program on its own group's questions
    calls = dict(T.MODEL_CALLS)
    run_cli(EV, args + ["--routed-only", "--baselines", "", "--out", str(TMP / "routed_champions"), "--live-cache",
                        str(run / "dev_eval" / next(f.name for f in (run / "dev_eval").glob("rounds_*.jsonl")))])
    c = json.loads((TMP / "routed_champions" / "results_k3.json").read_text())
    same = ("routed, held-out champions", "routed, strongest grid programs", "global champion")
    check(all(c["table"][k] == res["table"][k] for k in same) and dict(T.MODEL_CALLS) == calls
          and "held-out champions - strongest grid programs" in c["differences"]
          and not md_table_errors(TMP / "routed_champions" / "results_k3.md"),
          "routed only with the champions: the same routed and global scores, no model call")
    try:                                   # a call cap that stops step 9 part way stops it nonzero
        run_cli(EV, args + ["--out", str(TMP / "capped"), "--max-total-calls", "5"])
        check(False, "the call cap stops the evaluation nonzero")
    except SystemExit as exc:
        check("call cap stopped the evaluation" in str(exc), "the call cap stops the evaluation nonzero", str(exc))
    try:
        run_cli(EV, args + ["--visible-reasoning", "--out", str(TMP / "other")])
        check(False, "settings that differ from the search are refused")
    except SystemExit:
        check(True, "settings that differ from the search are refused")
    finally:
        P.SF.set_visible_reasoning(False)


def md_table_errors(path: Path) -> list[int]:
    """Lines of a markdown report whose table rows do not have the separator line's number of cells."""
    lines, bad = path.read_text().splitlines(), []
    for i, line in enumerate(lines):
        if line.startswith("|---"):
            j, n = i + 1, line.count("|")
            bad += [i - 1] if lines[i - 1].count("|") != n else []
            while j < len(lines) and lines[j].startswith("|"):
                bad += [j] if lines[j].count("|") != n else []
                j += 1
    return bad


def run_files(run: Path):
    for f in sorted(run.iterdir()):
        if f.is_file():
            yield f.name, f.read_bytes()


if __name__ == "__main__":
    P.configure_executor(executor="v3", window=65536)      # the entry scripts do the same
    T3.test_subsets()
    seeds_path = T3.test_seeds()
    routes = test_routes()
    test_eval(seeds_path, routes)
    print()
    if T.FAILURES:
        print(f"{len(T.FAILURES)} FAILED: " + "; ".join(T.FAILURES))
        sys.exit(1)
    print(f"all checks passed (model calls: {T.MODEL_CALLS}); temp dir {TMP}")

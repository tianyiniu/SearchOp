"""Offline portfolio + aggregator sweep over ALREADY-CACHED debate-arm answers.

Sizes the E4 (portfolio + aggregator coevolution) prize before spending any GPU
time. Reads nothing but files already in outputs/ -- no LLM calls, no API keys,
no vLLM, stdlib only. Typical runtime: under a minute.

THE QUESTION IT ANSWERS
    Individual schema accuracy on strict_test is saturated (best single arm =
    always_critic at 7.9%), but the union of the 6 arms' answer letters covers
    21.9% of questions. How much of that 21.9% is reachable by a DEPLOYABLE
    aggregator (one that never sees the gold letter), and at what call cost?

WHAT IT COMPUTES
  1. per-arm stats, including the departure decomposition
     accuracy = P(depart from base answer) x P(correct | departed),
     where "base answer" is the modal letter of the 7 cached minimal-schema
     samples in the best-of-n cache (label-free).
  2. the full grid: every non-empty subset of arms x every aggregator rule ->
     accuracy, mean calls, oracle coverage. IN-SAMPLE, for illumination only.
  3. a CROSS-VALIDATED estimate: repeated random split-half, config selected on
     the fit half (including the arm priority ordering and the vote weights),
     scored on the eval half. This is the honest number. With 63 subsets x ~9
     rules the in-sample maximum is badly selection-biased; do not quote it.
  4. the matched-CALL comparison that decides whether diversity beats
     resampling: at each call budget, the CV-selected heterogeneous portfolio vs
     single_pass sampled k times and aggregated by the SAME rules vs pass@k.

WHAT IT CANNOT ANSWER
    The homogeneous control here is single_pass only (1 call/sample, k<=7) --
    that is the only arm with repeated samples in outputs/. The control the
    portfolio claim really needs is always_critic best-of-k, which has never
    been run. Treat the homogeneous column as a lower bound on the resampling
    control, not as the control.

    python3 scripts/portfolio_sweep.py
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from statistics import mean, pstdev

LETTERS = "ABCDEFGHIJ"
DEFAULT_ARMS = "single_pass,self_critique,fixed_debate,always_critic,retrieve_copy,retrieve_synth"


# --- loading ---------------------------------------------------------------

def n_calls(schema: dict | None) -> int:
    """LLM calls one execution of this schema costs (one call per persona)."""
    if not schema or not schema.get("rounds"):
        return 1
    return sum(len(r.get("personas") or []) for r in schema["rounds"])


def load_arm(outdir: Path, arm: str, split: str) -> dict[str, dict]:
    path = outdir / f"retrieve_{arm}_{split}_cache.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        out[r["id"]] = {"letter": r.get("answer"), "correct": bool(r.get("correct")),
                        "calls": n_calls(r.get("schema"))}
    return out


def load_bestofn(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        out[r["id"]] = {"letters": r.get("letters") or [], "majority": r.get("majority"),
                        "n": r.get("n", 0), "n_correct": r.get("n_correct", 0),
                        "gold": r.get("answer_letter")}
    return out


def load_dataset(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text())
    out = {}
    for r in rows:
        opts = r["options"]
        if isinstance(opts, str):
            try:
                opts = ast.literal_eval(opts)
            except Exception:
                opts = []
        out[r["id"]] = {"n_options": max(2, len(opts)), "field": r.get("field"),
                        "difficulty": r.get("difficulty"), "gold": r.get("answer_letter")}
    return out


# --- aggregator rules ------------------------------------------------------
# A ballot is [(arm, letter), ...] in PRIORITY order (best arm first). Letters may
# be None (unparsed) and are ignored. Every rule is deployable: it never sees gold.

def _tally(ballot):
    return Counter(l for _, l in ballot if l)


def _by_priority(ballot, candidates):
    for _, l in ballot:
        if l in candidates:
            return l
    return None


def r_first(ballot, ctx):
    """Take the most reliable arm's letter. The within-subset single-arm floor."""
    for _, l in ballot:
        if l:
            return l
    return None


def r_plurality(ballot, ctx):
    t = _tally(ballot)
    if not t:
        return None
    m = max(t.values())
    return _by_priority(ballot, {l for l, c in t.items() if c == m})


def r_anti_plurality(ballot, ctx):
    """Least-frequent letter -- the inverted selector that scores 13.6% on 6 arms."""
    t = _tally(ballot)
    if not t:
        return None
    m = min(t.values())
    return _by_priority(ballot, {l for l, c in t.items() if c == m})


def r_singleton(ballot, ctx):
    """Highest-priority letter that exactly one arm produced; else plurality."""
    t = _tally(ballot)
    for _, l in ballot:
        if l and t[l] == 1:
            return l
    return r_plurality(ballot, ctx)


def r_weighted(ballot, ctx):
    w = ctx["weights"]
    s = defaultdict(float)
    for a, l in ballot:
        if l:
            s[l] += w.get(a, 1.0)
    if not s:
        return None
    m = max(s.values())
    return _by_priority(ballot, {l for l, v in s.items() if v == m})


def _drop_base(ballot, ctx):
    b = ctx.get("base")
    return [(a, l) for a, l in ballot if l and l != b]


def r_drop_base_first(ballot, ctx):
    kept = _drop_base(ballot, ctx)
    return r_first(kept, ctx) if kept else r_first(ballot, ctx)


def r_drop_base_plurality(ballot, ctx):
    kept = _drop_base(ballot, ctx)
    return r_plurality(kept, ctx) if kept else r_plurality(ballot, ctx)


def r_drop_base_anti(ballot, ctx):
    kept = _drop_base(ballot, ctx)
    return r_anti_plurality(kept, ctx) if kept else r_anti_plurality(ballot, ctx)


def r_flip_if_unanimous(ballot, ctx):
    """Top arm's letter, unless every arm agrees -- then take any other letter."""
    t = _tally(ballot)
    top = r_first(ballot, ctx)
    if len(t) == 1 and sum(t.values()) >= 2:
        alt = [l for _, l in ballot if l and l != top]
        return alt[0] if alt else top
    return top


RULES = {
    "first": r_first,
    "plurality": r_plurality,
    "weighted_plurality": r_weighted,
    "anti_plurality": r_anti_plurality,
    "singleton": r_singleton,
    "drop_base_first": r_drop_base_first,
    "drop_base_plurality": r_drop_base_plurality,
    "drop_base_anti": r_drop_base_anti,
    "flip_if_unanimous": r_flip_if_unanimous,
}


# --- grid evaluation -------------------------------------------------------

def rank_weights(perm: tuple[str, ...]) -> dict[str, float]:
    """Vote weights as a deterministic function of the priority permutation, so
    the whole grid caches on the permutation alone (keeps CV tractable)."""
    n = len(perm)
    return {a: float(n - i) for i, a in enumerate(perm)}


def eval_grid(perm, subsets, qids, arms, gold, base, rules):
    """{(subset, rule): (correct_vector, mean_calls)} for one priority ordering."""
    weights = rank_weights(perm)
    out = {}
    for subset in subsets:
        ordered = [a for a in perm if a in subset]           # priority order
        calls = mean(sum(arms[a][q]["calls"] for a in ordered) for q in qids)
        ballots = [[(a, arms[a][q]["letter"]) for a in ordered] for q in qids]
        for rname in rules:
            fn = RULES[rname]
            vec = []
            for q, ballot in zip(qids, ballots):
                ctx = {"weights": weights, "base": base.get(q)}
                vec.append(int(fn(ballot, ctx) == gold[q]))
            out[(subset, rname)] = (vec, calls)
    return out


def priority_from(acc_by_arm: dict[str, float]) -> tuple[str, ...]:
    return tuple(sorted(acc_by_arm, key=lambda a: (-acc_by_arm[a], a)))


def acc_on(vec, idx):
    return sum(vec[i] for i in idx) / len(idx) if idx else 0.0


# --- homogeneous (resampling) control --------------------------------------

def pass_at_k(n, c, k):
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def homogeneous_control(bon, qids, rules, kmax, repeats, seed):
    """single_pass sampled k times, aggregated by the SAME rules. Cost = k calls.
    For drop_base_* rules the base letter is taken from the HELD-OUT samples so
    the rule is not dropping the majority of the very ballots it is voting over;
    those rules are skipped at k=n_samples (no held-out sample left)."""
    rng = random.Random(seed)
    out = {}
    for k in range(1, kmax + 1):
        per_rule = {r: [] for r in rules}
        oracle = []
        for q in qids:
            b = bon[q]
            ls, n, g = b["letters"], len(b["letters"]), b["gold"]
            if n < k:
                continue
            oracle.append(pass_at_k(n, b["n_correct"], k))
            hits = {r: 0 for r in rules}
            for _ in range(repeats):
                idx = rng.sample(range(n), k)
                ballot = [(f"sp{j}", ls[j]) for j in idx]
                rest = [ls[j] for j in range(n) if j not in set(idx)]
                bt = Counter(l for l in rest if l)
                base_letter = bt.most_common(1)[0][0] if bt else None
                ctx = {"weights": {f"sp{j}": 1.0 for j in idx}, "base": base_letter}
                for r in rules:
                    if r.startswith("drop_base") and base_letter is None:
                        continue
                    hits[r] += int(RULES[r](ballot, ctx) == g)
            for r in rules:
                per_rule[r].append(hits[r] / repeats)
        out[k] = {"n": len(oracle), "pass_at_k": mean(oracle) if oracle else 0.0,
                  "rules": {r: (mean(v) if v else 0.0) for r, v in per_rule.items()}}
    return out


# --- main ------------------------------------------------------------------

def main(args):
    arms_wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = {}
    for a in arms_wanted:
        d = load_arm(args.outdir, a, args.split)
        if d:
            arms[a] = d
        else:
            print(f"  ! no cache for arm {a!r} (retrieve_{a}_{args.split}_cache.jsonl) -- skipping")
    if len(arms) < 2:
        raise SystemExit("need at least 2 arms with caches")

    meta = load_dataset(args.dataset)
    bon = load_bestofn(args.bestofn_cache)
    qids = sorted(set.intersection(*[set(d) for d in arms.values()]) & set(meta) & set(bon))
    if not qids:
        raise SystemExit("no questions common to all arm caches, the dataset, and the best-of-n cache")
    gold = {q: meta[q]["gold"] for q in qids}
    base = {q: bon[q]["majority"] for q in qids}      # label-free base-model answer
    rules = list(RULES)
    names = sorted(arms)
    print(f"split={args.split}  arms={names}  questions={len(qids)}  "
          f"rules={len(rules)}  base-letter source=bestofn majority\n")

    # --- 1. per-arm stats, departure-decomposed ---------------------------
    print(f"{'arm':<16}{'acc':>8}{'calls':>8}{'depart':>9}{'prec|dep':>10}{'prec|stay':>11}")
    per_arm = {}
    for a in names:
        d = arms[a]
        acc = mean(d[q]["correct"] for q in qids)
        calls = mean(d[q]["calls"] for q in qids)
        dep = [q for q in qids if d[q]["letter"] is not None and d[q]["letter"] != base[q]]
        stay = [q for q in qids if q not in set(dep)]
        pdep = mean(d[q]["correct"] for q in dep) if dep else 0.0
        pstay = mean(d[q]["correct"] for q in stay) if stay else 0.0
        per_arm[a] = {"accuracy": acc, "mean_calls": calls, "departure_rate": len(dep) / len(qids),
                      "precision_given_departure": pdep, "precision_given_stay": pstay}
        print(f"{a:<16}{acc:>8.1%}{calls:>8.1f}{len(dep)/len(qids):>9.1%}{pdep:>10.1%}{pstay:>11.1%}")
    null = mean(1.0 / (meta[q]["n_options"] - 1) for q in qids)
    print(f"\n  random-departure null (uniform over the other options): {null:.1%}")

    # --- 2. full in-sample grid -------------------------------------------
    subsets = [tuple(c) for r in range(1, min(args.max_portfolio, len(names)) + 1)
               for c in combinations(names, r)]
    full_perm = priority_from({a: per_arm[a]["accuracy"] for a in names})
    grid = eval_grid(full_perm, subsets, qids, arms, gold, base, rules)
    all_idx = list(range(len(qids)))

    oracle_cov = {s: mean(any(arms[a][q]["correct"] for a in s) for q in qids) for s in subsets}
    rowsq = [{"subset": s, "rule": r, "accuracy": acc_on(v, all_idx), "mean_calls": c,
              "oracle_coverage": oracle_cov[s]} for (s, r), (v, c) in grid.items()]
    rowsq.sort(key=lambda x: -x["accuracy"])
    print(f"\n--- top {args.top} of {len(rowsq)} (subset x rule) configs, IN-SAMPLE "
          f"(selection-biased -- see the CV table below) ---")
    print(f"{'accuracy':>9}{'calls':>7}{'oracle':>8}  {'rule':<21}subset")
    for x in rowsq[:args.top]:
        print(f"{x['accuracy']:>9.1%}{x['mean_calls']:>7.1f}{x['oracle_coverage']:>8.1%}  "
              f"{x['rule']:<21}{'+'.join(x['subset'])}")

    print("\n--- in-sample Pareto front on (mean calls, accuracy) ---")
    pareto, best_so_far = [], -1.0
    for x in sorted(rowsq, key=lambda x: (x["mean_calls"], -x["accuracy"])):
        if x["accuracy"] > best_so_far:
            best_so_far = x["accuracy"]
            pareto.append(x)
            print(f"{x['accuracy']:>9.1%}{x['mean_calls']:>7.1f}{x['oracle_coverage']:>8.1%}  "
                  f"{x['rule']:<21}{'+'.join(x['subset'])}")

    # --- 3. cross-validated selection -------------------------------------
    caps = [float(c) for c in args.budgets.split(",")]
    rng = random.Random(args.seed)
    cache = {full_perm: grid}
    cv = {c: {"scores": [], "picked": Counter()} for c in caps}
    for _ in range(args.cv_repeats):
        order = all_idx[:]
        rng.shuffle(order)
        half = len(order) // 2
        fit, ev = order[:half], order[half:]
        perm = priority_from({a: acc_on([arms[a][qids[i]]["correct"] for i in all_idx], fit)
                              for a in names})
        g = cache.get(perm)
        if g is None:
            g = cache[perm] = eval_grid(perm, subsets, qids, arms, gold, base, rules)
        for cap in caps:
            elig = [(k, v) for k, v in g.items() if v[1] <= cap]
            if not elig:
                continue
            bk, bv = max(elig, key=lambda kv: acc_on(kv[1][0], fit))
            cv[cap]["scores"].append(acc_on(bv[0], ev))
            cv[cap]["picked"][bk] += 1

    # --- 4. matched-call comparison ---------------------------------------
    homog = homogeneous_control(bon, qids, rules, args.homog_kmax, args.homog_repeats, args.seed)
    print(f"\n--- HONEST comparison at matched CALL budget "
          f"({args.cv_repeats} split-half repeats; portfolio config chosen on the fit half) ---")
    print(f"{'calls':>6}{'portfolio CV':>14}{'(sd)':>7}{'single_pass@k':>15}"
          f"{'best homog rule':>18}{'pass@k':>9}  modal portfolio config")
    for cap in caps:
        s = cv[cap]["scores"]
        if not s:
            continue
        k = min(args.homog_kmax, int(cap))
        h = homog.get(k)
        hbest = max(h["rules"].items(), key=lambda kv: kv[1]) if h else ("-", 0.0)
        pick = cv[cap]["picked"].most_common(1)[0]
        frac = pick[1] / max(1, sum(cv[cap]["picked"].values()))
        print(f"{cap:>6.0f}{mean(s):>14.1%}{pstdev(s):>7.1%}"
              f"{(hbest[1] if h else 0.0):>15.1%}{hbest[0]:>18}"
              f"{(h['pass_at_k'] if h else 0.0):>9.1%}  "
              f"{pick[0][1]}:{'+'.join(pick[0][0])} ({frac:.0%})")
    print("\n  single_pass@k is the ONLY resampling control available offline (1 call/sample,"
          f" k<={args.homog_kmax}).\n  The control the portfolio claim needs is always_critic"
          " best-of-k, which has never been run.")

    # --- 5. stratified view of the best CV config --------------------------
    top_cap = caps[-1]
    if cv[top_cap]["picked"]:
        cfg = cv[top_cap]["picked"].most_common(1)[0][0]
        vec = grid[cfg][0]
        print(f"\n--- modal config at <={top_cap:.0f} calls: {cfg[1]} over {'+'.join(cfg[0])} ---")
        for key in ("difficulty", "field"):
            groups = defaultdict(list)
            for i, q in enumerate(qids):
                groups[meta[q][key]].append(i)
            print(f"  by {key}:")
            for gname, idx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                if len(idx) < args.min_stratum:
                    continue
                print(f"    {str(gname):<34} n={len(idx):>4}  {acc_on(vec, idx):>6.1%}")

    # --- write ------------------------------------------------------------
    out = {
        "split": args.split, "n_questions": len(qids), "arms": names,
        "random_departure_null": null,
        "per_arm": per_arm,
        "in_sample_top": [{**x, "subset": list(x["subset"])} for x in rowsq[:args.top]],
        "in_sample_pareto": [{**x, "subset": list(x["subset"])} for x in pareto],
        "cross_validated": {
            str(c): {"mean": mean(v["scores"]) if v["scores"] else None,
                     "sd": pstdev(v["scores"]) if v["scores"] else None,
                     "n_repeats": len(v["scores"]),
                     "modal_config": (lambda p: {"subset": list(p[0][0]), "rule": p[0][1],
                                                 "share": p[1] / max(1, sum(v["picked"].values()))})
                                     (v["picked"].most_common(1)[0]) if v["picked"] else None}
            for c, v in cv.items()},
        "homogeneous_single_pass": {str(k): v for k, v in homog.items()},
        "caveat": "in_sample_* are selection-biased over %d configs; quote cross_validated. "
                  "The homogeneous control is single_pass only; always_critic best-of-k is missing."
                  % len(rowsq),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--split", default="strict_test", help="Split tag in retrieve_<arm>_<split>_cache.jsonl")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_strict_test.json"))
    ap.add_argument("--bestofn-cache", type=Path, default=Path("outputs/bestofn_strict_cache.jsonl"),
                    help="Supplies the label-free base letter and the resampling control.")
    ap.add_argument("--max-portfolio", type=int, default=6, help="Largest subset size to enumerate.")
    ap.add_argument("--budgets", default="1,2,3,4,5,6,7,8,10,12,16",
                    help="Call-budget caps for the cross-validated comparison.")
    ap.add_argument("--cv-repeats", type=int, default=200, help="Split-half repeats.")
    ap.add_argument("--homog-kmax", type=int, default=7, help="Max k for the single_pass control.")
    ap.add_argument("--homog-repeats", type=int, default=50, help="Subsamples per (question,k).")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-stratum", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/portfolio_sweep_strict_test.json"))
    main(ap.parse_args())

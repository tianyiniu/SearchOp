"""The external baselines on the TRAINING search questions, beside the seeds.

Generation 0 of the search scores every seed on every search question (two
replicates). This puts the independently run baselines (baselines/generate.py
and baselines/selfrefine.py, full thinking budget) on the same questions, so
the search's starting point can be read against them before any dev question
is touched.

  1. export the search questions as a dataset the baseline scripts can read:
       python scripts/train_baselines.py --export --clusters <clusters.json> --per-group 50 \\
           --out datasets/search_questions.json
  2. run baselines/generate.py and baselines/selfrefine.py on it, then baselines/score.py --save
  3. compare, once generation 0 has run:
       python scripts/train_baselines.py --compare --run outputs/<search>/run1 \\
           --external "direct=baselines/results/direct_<tag>_train_k3.json,self-refine=..."

Writes <run>/train_baselines.md and .json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import program_space as P  # noqa: E402
from program_space import ProgRecord  # noqa: E402

ROOT = P.ROOT


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else 0.0)


def export(args) -> None:
    groups = P.load_groups(args.clusters, args.per_group, args.dev_split)
    qids = [q for g in groups["groups"] for q in g["search"]]
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    missing = [q for q in qids if q not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} search questions are not in {args.dataset}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([rows[q] for q in qids], ensure_ascii=False, indent=1))
    print(f"{len(qids)} search questions ({groups['k']} groups, "
          + ("every question" if args.per_group == 0 else f"at most {args.per_group} each") + f") -> {args.out}")


def compare(args) -> None:
    lines = [l for l in (args.run / "archive.jsonl").open() if l.strip()]
    header = json.loads(lines[0])
    recs: dict[str, dict] = {}
    for l in lines[1:]:
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue
        recs[d["key"]] = d
    qids = header["qids"]
    groups = {g: qs for g, qs in header["groups"].items()}
    seeds = [ProgRecord.from_json(d) for d in recs.values() if d["gen"] == 0]
    seeds = [s for s in seeds if not s.gaps(qids, 0)]
    if not seeds:
        raise SystemExit(f"no seed in {args.run} is scored on every search question yet")

    per_q: dict[str, list[float]] = {}                  # row -> per question mean mark
    info: dict[str, dict] = {}
    for s in seeds:
        per_q[s.name] = [s.mark(q) or 0.0 for q in qids]
        info[s.name] = {"source": header.get("seed_source", {}).get(s.name, "?"),
                        "turns": s.turns(qids), "tokens": s.tokens_used(qids),
                        "reps": sorted(s.reps)}
    externals = []
    for pair in [x for x in args.external.split(",") if x]:
        label, path = pair.split("=", 1)
        pq = json.loads(Path(path).read_text())["per_question"]
        lacking = [q for q in qids if q not in pq]
        if lacking:
            raise SystemExit(f"--external {label}: {len(lacking)} search questions are not in {path}")
        name = f"external {label}"
        externals.append(name)
        # right/wrong per run: the scorer's marks (a judge's verdicts for open answers), or letter == key
        marks = {q: (pq[q]["marks"] if "marks" in pq[q] else [int(p == pq[q]["answer"]) for p in pq[q]["preds"]])
                 for q in qids}
        per_q[name] = [sum(marks[q]) / len(marks[q]) for q in qids]
        toks = [sum(pq[q]["tokens"]) / len(pq[q]["tokens"]) for q in qids if pq[q].get("tokens")]
        info[name] = {"source": "external", "turns": None, "tokens": sum(toks) / len(toks) if toks else None,
                      "runs": min(len(pq[q]["preds"]) for q in qids)}

    rows = sorted(per_q, key=lambda k: -sum(per_q[k]))
    table = {}
    out = [f"# Seeds vs external baselines on the {len(qids)} training search questions", "",
           f"Run {args.run}. A seed's accuracy is the mean over questions of its mean mark over its "
           f"replicates (generation 0 scores two). An external baseline's is its avg@k over its runs. "
           f"Differences are paired over questions, with their standard error.", "",
           "| row | source | acc | ± | turns | tokens | " + " | ".join(f"minus {e}" for e in externals) + " |",
           "|---|---|---|---|---|---|" + "---|" * len(externals)]
    for name in rows:
        m, se = mean_se(per_q[name])
        diffs = {}
        for e in externals:
            d, dse = mean_se([a - b for a, b in zip(per_q[name], per_q[e])])
            diffs[e] = {"diff": d, "se": dse}
        table[name] = {"acc": m, "se": se, **info[name], "vs": diffs}
        i = info[name]
        out.append(f"| {name} | {i['source']} | {m:.1%} | {se:.1%} | "
                   + ("-" if i["turns"] is None else f"{i['turns']:.1f}") + " | "
                   + ("-" if i["tokens"] is None else f"{i['tokens']:,.0f}") + " | "
                   + " | ".join(f"{diffs[e]['diff']:+.1%} ± {diffs[e]['se']:.1%}" for e in externals) + " |")
    # per group: the best seed there against the externals there
    out += ["", "Per group: the best seed on the group (picked on these same questions, so read it as "
                "optimistic) and the externals.", "",
            "| group | questions | best seed | its acc | " + " | ".join(externals) + " |",
            "|---|---|---|---|" + "---|" * len(externals)]
    idx = {q: i for i, q in enumerate(qids)}
    per_group = {}
    for g, qs in groups.items():
        acc = {name: sum(per_q[name][idx[q]] for q in qs) / len(qs) for name in per_q}
        best = max((s.name for s in seeds), key=lambda n: acc[n])
        per_group[g] = {"best_seed": best, "acc": acc}
        out.append(f"| {g} | {len(qs)} | {best} | {acc[best]:.1%} | "
                   + " | ".join(f"{acc[e]:.1%}" for e in externals) + " |")
    (args.run / "train_baselines.json").write_text(json.dumps({"run": str(args.run), "n_questions": len(qids),
                                                                "table": table, "per_group": per_group}, indent=1))
    (args.run / "train_baselines.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n-> {args.run / 'train_baselines.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true", help="write the search questions as a dataset")
    mode.add_argument("--compare", action="store_true", help="seeds vs external baselines on those questions")
    ap.add_argument("--clusters", type=Path, default=ROOT / "outputs/describe_v3/clusters_600_train.json")
    ap.add_argument("--per-group", type=int, default=50)
    ap.add_argument("--dev-split", type=Path, default=None,
                    help="a split file of split_train_dev.py: each group's dev questions are held out for "
                         "--pick-champions, the rest are search questions (with --per-group N > 0, the "
                         "first N of them)")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_600_train.json")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/search_questions.json")
    ap.add_argument("--run", type=Path, help="search run directory (--compare)")
    ap.add_argument("--external", default="", help="label=score-file pairs, comma separated (baselines/score.py "
                                                   "--save on the exported questions)")
    args = ap.parse_args()
    if args.export:
        export(args)
    else:
        if args.run is None:
            raise SystemExit("--compare needs --run")
        compare(args)


if __name__ == "__main__":
    main()

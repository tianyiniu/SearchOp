"""How much would a perfect per-question config chooser win on FRAMES?

Reads every outputs/search_*_cache.jsonl, takes each config's final correctness
per question, and reports:

  - per-config accuracy on its own questions
  - accuracy and union ("any config correct") on the common question subset
  - the same union for answer-in-evidence
  - how many questions are solved by exactly one config

The union is an oracle number: the score of a chooser that always picks the
right config. No model calls; reads only existing caches.

    python3 scripts/frames_headroom.py
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs"
MIN_N = 200          # skip partial/debug caches


def load(path: Path) -> dict[str, dict]:
    """Last record per question id (later rows supersede earlier reruns)."""
    rows: dict[str, dict] = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("id"):
            rows[r["id"]] = r
    return rows


def ok(r: dict) -> bool:
    return r.get("status", "ok") == "ok" and r.get("downstream_correct") is not None


def main() -> None:
    configs: dict[str, dict[str, dict]] = {}
    for path in sorted(OUT.glob("search_*_cache.jsonl")):
        rows = {qid: r for qid, r in load(path).items() if ok(r)}
        name = path.stem.removeprefix("search_").removesuffix("_cache")
        if len(rows) >= MIN_N:
            configs[name] = rows
        else:
            print(f"skipping {name}: only {len(rows)} usable rows")

    common = set.intersection(*(set(v) for v in configs.values()))
    print(f"\n{len(configs)} configs, {len(common)} questions with a usable "
          f"result in every config\n")

    print(f"{'config':34}{'n':>5}{'acc':>8}{'acc(common)':>13}{'ans-in-evid':>13}")
    for name, rows in sorted(configs.items(),
                             key=lambda kv: -sum(r['downstream_correct'] for r in kv[1].values()) / len(kv[1])):
        acc = sum(r["downstream_correct"] for r in rows.values()) / len(rows)
        acc_c = sum(rows[q]["downstream_correct"] for q in common) / len(common)
        aie = sum(bool(rows[q].get("answer_in_evidence")) for q in common) / len(common)
        print(f"{name:34}{len(rows):>5}{acc:>8.1%}{acc_c:>13.1%}{aie:>13.1%}")

    solved_by = {q: [c for c, rows in configs.items() if rows[q]["downstream_correct"]]
                 for q in common}
    n_any = sum(1 for v in solved_by.values() if v)
    n_one = sum(1 for v in solved_by.values() if len(v) == 1)
    aie_any = sum(1 for q in common
                  if any(bool(rows[q].get("answer_in_evidence")) for rows in configs.values()))
    best_single = max(sum(rows[q]["downstream_correct"] for q in common) / len(common)
                      for rows in configs.values())

    print(f"\nbest single config (common subset): {best_single:.1%}")
    print(f"union, any config correct:          {n_any / len(common):.1%}  ({n_any}/{len(common)})")
    print(f"  solved by exactly one config:     {n_one}  "
          f"({n_one / max(n_any, 1):.0%} of solved)")
    print(f"union, answer in evidence:          {aie_any / len(common):.1%}")
    print(f"headroom for a per-question chooser: "
          f"{n_any / len(common) - best_single:+.1%}")

    counts = Counter(len(v) for v in solved_by.values())
    print("\nquestions by number of configs that solve them:")
    for k in sorted(counts):
        print(f"  {k:>2} configs: {counts[k]}")

    print("\nsole solver counts (questions only this config gets right):")
    sole = Counter(v[0] for v in solved_by.values() if len(v) == 1)
    for name, c in sole.most_common():
        print(f"  {name:34}{c}")

    print("\nbest 2-config pairs (union on common subset):")
    pairs = sorted(((sum(1 for q in common
                         if configs[a][q]["downstream_correct"]
                         or configs[b][q]["downstream_correct"]) / len(common), a, b)
                    for a, b in combinations(configs, 2)), reverse=True)[:5]
    for u, a, b in pairs:
        print(f"  {u:.1%}  {a} + {b}")


if __name__ == "__main__":
    main()

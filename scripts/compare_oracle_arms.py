"""Paired comparison of the oracle-ceiling arms (scripts/oracle_search_ceiling.py).

Arm accuracies are computed over slightly different question sets (a fetch can
fail, a cached context can be missing), and the arms are highly correlated —
they answer the SAME questions from differently packaged evidence. So a
difference of a few points between two marginal rates says very little. This
reports the paired view instead:

  - every rate recomputed on the subset where BOTH arms produced an answer
  - the discordant counts (b = A right & B wrong, c = A wrong & B right), which
    are what a difference actually rests on
  - an exact McNemar p-value on those discordant pairs (binomial, two-sided)

    $PY scripts/compare_oracle_arms.py
    $PY scripts/compare_oracle_arms.py --baseline raw_pack_fix --metric correct_visible
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_arm(out_dir: Path, arm: str) -> dict[str, dict]:
    path = out_dir / f"oracle_{arm}_cache.jsonl"
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") == "ok":
            records[rec["id"]] = rec
    return records


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main(args: argparse.Namespace) -> None:
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    data = {a: load_arm(args.out_dir, a) for a in arms}
    data = {a: d for a, d in data.items() if d}
    if args.baseline not in data:
        raise SystemExit(f"baseline arm {args.baseline!r} has no records in {args.out_dir}")

    print(f"metric: {args.metric}\n")
    print(f"{'arm':<16}{'n':>6}{'accuracy':>11}   (marginal, own subset)")
    for arm, recs in data.items():
        n = len(recs)
        acc = sum(1 for r in recs.values() if r.get(args.metric)) / n if n else 0.0
        print(f"{arm:<16}{n:>6}{acc:>11.1%}")

    base = data[args.baseline]
    print(f"\npaired vs {args.baseline}:")
    print(f"{'arm':<16}{'n_pair':>8}{'base':>9}{'arm':>9}{'delta':>9}"
          f"{'base_only':>11}{'arm_only':>10}{'p':>9}")
    for arm, recs in data.items():
        if arm == args.baseline:
            continue
        ids = sorted(set(base) & set(recs))
        if not ids:
            continue
        b = sum(1 for i in ids if base[i].get(args.metric) and not recs[i].get(args.metric))
        c = sum(1 for i in ids if not base[i].get(args.metric) and recs[i].get(args.metric))
        acc_b = sum(1 for i in ids if base[i].get(args.metric)) / len(ids)
        acc_a = sum(1 for i in ids if recs[i].get(args.metric)) / len(ids)
        print(f"{arm:<16}{len(ids):>8}{acc_b:>9.1%}{acc_a:>9.1%}{(acc_a - acc_b) * 100:>+9.1f}"
              f"{b:>11}{c:>10}{mcnemar_exact(b, c):>9.3f}")
    print("\ndelta is in percentage points; base_only/arm_only are the discordant pairs.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--arms", default="raw_pack,raw_pack_fix,head_small,passages,passages_big,cached_ctx")
    ap.add_argument("--baseline", default="raw_pack_fix")
    ap.add_argument("--metric", default="correct_visible",
                    choices=["correct_visible", "correct_raw", "answer_in_context", "refused"])
    main(ap.parse_args())

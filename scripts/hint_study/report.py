"""The funnel and the tables, from whatever study files exist so far.

    python scripts/hint_study/report.py            # writes <out-dir>/report.md and prints it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def pct(x) -> str:
    return "-" if x is None or x != x else f"{100 * x:.1f}%"


def funnel(out_dir: Path, sheets_dir: Path, split: str, rows: list[dict]) -> list[str]:
    un = C.read_json(out_dir / f"unaided_{split}.json")
    sheets = C.read_sheets(sheets_dir)
    leaks = C.read_json(out_dir / "leak.json", {})
    active = C.active_sheets(sheets_dir, out_dir)
    mid = C.read_json(out_dir / f"middle_{split}.json")
    lines = [f"### {split}: the funnel", "", "| stage | all | recall | derive | both |", "|---|---|---|---|---|"]

    def row(label, pred):
        cells = [str(sum(pred(r) for r in rs)) for rs in [rows] + [C.by_kind(rows)[k] for k in C.KINDS]]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("drawn", lambda r: True)
    if un:
        pq = un["per_question"]
        row(f"fails alone (right at most {C.FAIL_MAX} of {un['k']})", lambda r: bool(pq.get(r["id"], {}).get("fails_alone")))
    row("sheet written", lambda r: r["id"] in sheets)
    row("teacher agrees with the key", lambda r: any(v["teacher_agrees"] for v in sheets.get(r["id"], [])))
    row("sheet checked for leaks", lambda r: any(f"{r['id']}|{v['version']}" in leaks for v in sheets.get(r["id"], [])))
    row("every version leaked", lambda r: bool(sheets.get(r["id"])) and all(
        leaks.get(f"{r['id']}|{v['version']}", {}).get("leak") for v in sheets[r["id"]]))
    row("usable sheet", lambda r: r["id"] in active)
    if mid:
        pq = mid["per_question"]
        row(f"middle (right at least {C.PASS_MIN} of {mid['k']} with the sheet)", lambda r: bool(pq.get(r["id"], {}).get("keep")))
    lines.append("")
    if un:
        lines += [f"Alone: right {pct(un['summary']['all'][f'avg@{un['k']}'])} of tries, right at least once on "
                  f"{pct(un['summary']['all'][f'pass@{un['k']}'])} of questions, "
                  f"{pct(un['summary']['all']['truncated_share'])} of replies ran out of room, "
                  f"{un['summary']['all']['tokens_per_reply']:.0f} tokens per reply.", ""]
    if mid and mid.get("trimmed"):
        s = mid["summary"].get("all", {})
        lines += [f"Sheets of middle questions: {s.get('items_mean', 0):.1f} items on average, "
                  f"{s.get('min_items_mean', 0):.1f} after trimming.", ""]
    return lines


def hints_table(out_dir: Path, split: str) -> list[str]:
    hn = C.read_json(out_dir / f"hints_needed_{split}.json")
    if not hn:
        return []
    lines = [f"### {split}: hints needed by way of working (K = {hn['k']}, finishes = right {hn['reach_min']}+ times)", "",
             "| way of working | questions | cannot finish even with all | needs no hints | items needed (mean) | share of sheet |",
             "|---|---|---|---|---|---|"]
    for m, by in hn["summary"].items():
        s = by["all"]
        lines.append(f"| {m} | {s['questions']} | {s['cannot_finish']} | {s['finishes_with_no_hints']} | "
                     f"{s['needed_mean']:.2f} | {pct(s['share_of_sheet_mean'])} |")
    lines.append("")
    if hn.get("continue_high_curve"):
        lines += ["Plain 'just continue', how often right against how many hints given:", "",
                  "| hints given | questions | right |", "|---|---|---|"]
        for kk, v in hn["continue_high_curve"].items():
            lines.append(f"| {kk} | {v['questions']} | {pct(v['pass_rate'])} |")
        lines.append("")
    return lines


def facts_table(out_dir: Path, split: str) -> list[str]:
    fc = C.read_json(out_dir / f"facts_{split}.json")
    if not fc:
        return []
    lines = [f"### {split}: which facts the model has", "",
             "| kind | questions | facts | can state | recognises only | does not have | questions needing a lookup |",
             "|---|---|---|---|---|---|---|"]
    for kind, s in fc["summary"].items():
        f = s["fact_classes"]
        lines.append(f"| {kind} | {s['questions']} | {s['facts']} | {f['can_state']} | {f['recognises_only']} | "
                     f"{f['does_not_have']} | {s['question_buckets']['does_not_have']} |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    C.add_dir_args(ap)
    args = ap.parse_args()
    un = C.read_json(args.out_dir / "unaided_train.json") or C.read_json(args.out_dir / "unaided_test.json") or {}
    lines = [f"# Hint-sheet study, current numbers ({un.get('model', 'small model not recorded')})", ""]
    for split in ("train", "test"):
        try:
            rows = C.load_split(split)
        except SystemExit:
            continue
        lines += funnel(args.out_dir, C.teacher_dir(args), split, rows)
        lines += hints_table(args.out_dir, split)
        lines += facts_table(args.out_dir, split)
    text = "\n".join(lines)
    (args.out_dir / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

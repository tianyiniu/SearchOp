"""Score generate.py output: avg@k, pass@k, broken down by difficulty and discipline.

Answer extraction reproduces the official SuperGPQA zero-shot rule: letter patterns on the
last line, then on the whole response, then the same cascade matching option text.

GPQA-Diamond and MATH rows (a "dataset" field) are read and graded by tasks.py: the same letter
rule with only A-D, and math's last \\boxed{} answer compared with the key by math-verify.

HLE (rows with an "answer_type", scripts/prepare_hle.py): the answer is the response's last
'Exact Answer:' / 'Answer:' line (hle_format.extract), graded by the judge model with the
verdict cache the debate search uses (scripts/judge_answers.py). Every question's per-sample
right/wrong list is saved as "marks" (for SuperGPQA: letter == key).
"""
import argparse
import json
import re
import signal
from collections import defaultdict
from math import comb

import hle_format

WRAP = r"(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*"
TAIL = r"(?:\\?\}?\$?\)?\]?\}?)*"
END = r"(?:[\s:\.\*)]|$)"
LETTERS = "ABCDEFGHIJ"



def label_patterns(letters):
    """The letter patterns of the SuperGPQA rule, for the given choice letters."""
    return [
        rf"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*{WRAP}\s*([{letters}]){TAIL}{END}",
        rf"(?i:Answer)[\*\s]*:\s*{WRAP}\s*([{letters}]){TAIL}{END}",
        rf"^[^\w\r\n]*{WRAP}\s*([{letters}]){TAIL}{END}",
    ]


LABEL_PATTERNS = label_patterns(LETTERS)
CONTENT_WRAP = r"(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*"
CONTENT_PATTERNS = [
    r"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*" + CONTENT_WRAP + r"\s*({opts})" + TAIL + END,
    r"(?i:Answer)\s*" + CONTENT_WRAP + r"\s*({opts})" + TAIL + END,
    r"^[^\w\r\n]*" + CONTENT_WRAP + r"\s*({opts})" + TAIL + END,
]


class RegexTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise RegexTimeout


def search(pattern, text, flags=0, timeout=5):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        return re.search(pattern, text, flags)
    except (RegexTimeout, re.error):
        return None
    finally:
        signal.alarm(0)


def extract_letter(text, letters=LETTERS):
    """SuperGPQA's letter rule; `letters` narrows the letters it accepts (GPQA: A-D)."""
    patterns = LABEL_PATTERNS if letters == LETTERS else label_patterns(letters)
    text = text.rstrip()
    last_line = text.split("\n")[-1]
    for scope in (last_line, text):
        for pat in patterns:
            m = search(pat, scope, re.IGNORECASE)
            if m:
                return m.group(1)
    return None


def extract_by_content(text, options):
    escaped = [re.escape(o) for o in options]
    alternation = "|".join(escaped)
    text = text.rstrip()
    last_line = text.split("\n")[-1]
    for scope in (last_line, text):
        for pat in CONTENT_PATTERNS:
            m = search(pat.replace("{opts}", alternation), scope)
            if m:
                hit = m.group(1)
                return options[escaped.index(hit)] if hit in escaped else hit
    return None


def extract_answer(content, options, letters=LETTERS):
    if not isinstance(content, str):
        return None
    pred = extract_letter(content, letters)
    if pred is None:
        opt = extract_by_content(content, options)
        pred = chr(options.index(opt) + 65) if opt in options else None
    return pred


def pass_at_k(n, c, k):
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarize(rows, k):
    """rows: list of (n, c). avg@k is the mean per-sample accuracy; pass@j uses the unbiased estimator."""
    full = [(n, c) for n, c in rows if n >= k]
    if not full:
        return {"questions": 0}
    out = {"questions": len(full), f"avg@{k}": sum(c / n for n, c in full) / len(full)}
    j = 1
    while j <= k:
        out[f"pass@{j}"] = sum(pass_at_k(n, c, j) for n, c in full) / len(full)
        j *= 2
    if k not in [2**i for i in range(8)]:
        out[f"pass@{k}"] = sum(pass_at_k(n, c, k) for n, c in full) / len(full)
    return out


def main(args):
    with open(args.data) as f:
        items = {it["id"]: it for it in json.load(f)}

    samples = defaultdict(dict)
    with open(args.results) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None and r["id"] in items:
                samples[r["id"]][r["sample_idx"]] = r  # last write wins on duplicates

    chosen = {}
    for qid, by_idx in samples.items():
        if args.all_samples:
            chosen[qid] = [by_idx[s] for s in sorted(by_idx)]
        elif args.any_k:
            chosen[qid] = [by_idx[s] for s in sorted(by_idx)][: args.k]
        else:
            chosen[qid] = [by_idx[s] for s in sorted(by_idx) if s < args.k]
    hle = any(hle_format.is_hle(it) for it in items.values())
    if hle:
        from judge_answers import JUDGE_MODEL, Judge      # scripts/ is on the path (hle_format)
        judge = Judge(args.judge_cache, model=args.judge_model or JUDGE_MODEL)
        preds_of = {qid: [hle_format.extract(r["content"]) for r in recs] for qid, recs in chosen.items()}
        pairs = sorted({(qid, p) for qid, ps in preds_of.items() for p in ps if p is not None})
        verdict = dict(zip(pairs, judge.grade_many([(items[q], p) for q, p in pairs], workers=args.judge_workers)))
        print(f"judge {judge.model}: {len(pairs)} distinct answers; {judge.stats}")
    else:
        import tasks                    # GPQA's letters, math's \boxed answer (tasks imports this module)
        preds_of = {qid: [tasks.answer_of(r["content"], items[qid]) for r in recs]
                    for qid, recs in chosen.items()}

    n_samples = n_miss = n_trunc = tok = 0
    per_q = {}
    for qid, recs in chosen.items():
        it = items[qid]
        preds = preds_of[qid]
        gold = it["answer"] if hle or "answer_letter" not in it else it["answer_letter"]
        if hle:
            correct = [p is not None and verdict[(qid, p)] for p in preds]
        else:
            correct = [p is not None and tasks.correct(it, p) for p in preds]
        per_q[qid] = {"n": len(recs), "c": sum(correct), "preds": preds, "marks": [int(c) for c in correct],
                      "answer": gold, "difficulty": it["difficulty"],
                      "discipline": it["discipline"],
                      "tokens": [r["completion_tokens"] for r in recs]}
        n_samples += len(recs)
        n_miss += sum(p is None for p in preds)
        n_trunc += sum(r["finish_reason"] == "length" for r in recs)
        tok += sum(r["completion_tokens"] for r in recs)

    incomplete = sum(1 for q in per_q.values() if q["n"] < args.k)
    missing = len(items) - len(per_q)

    report = {
        "overall": summarize([(q["n"], q["c"]) for q in per_q.values()], args.k),
        "samples": n_samples,
        "miss_rate": n_miss / max(n_samples, 1),
        "truncation_rate": n_trunc / max(n_samples, 1),
        "avg_completion_tokens": tok / max(n_samples, 1),
        "questions_incomplete": incomplete,
        "questions_missing": missing,
    }
    for key in ("difficulty", "discipline"):
        groups = defaultdict(list)
        for q in per_q.values():
            groups[q[key]].append((q["n"], q["c"]))
        report[f"by_{key}"] = {g: summarize(v, args.k) for g, v in sorted(groups.items())}

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"report": report, "per_question": per_q}, f, indent=1, ensure_ascii=False)

    k = args.k
    o = report["overall"]
    print(f"\n=== SuperGPQA subset | {o.get('questions', 0)} questions with {k} samples ===")
    if incomplete or missing:
        print(f"  (excluded: {incomplete} incomplete, {missing} not started)")
    if o.get("questions"):
        print(f"  avg@{k}  = {100 * o[f'avg@{k}']:.2f}")
        print(f"  pass@{k} = {100 * o[f'pass@{k}']:.2f}")
    print(f"  miss rate {100 * report['miss_rate']:.2f}% | truncated {100 * report['truncation_rate']:.2f}% "
          f"| avg completion tokens {report['avg_completion_tokens']:.0f}")
    for key in ("difficulty", "discipline"):
        print(f"\n  {key:<22} {'n':>5} {f'avg@{k}':>8} {f'pass@{k}':>8}")
        for g, s in report[f"by_{key}"].items():
            if s.get("questions"):
                print(f"  {g:<22} {s['questions']:>5} {100 * s[f'avg@{k}']:>8.2f} {100 * s[f'pass@{k}']:>8.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--any-k", action="store_true",
                   help="use the first k finished samples of each question (any index) instead of "
                        "indices 0..k-1; biased toward shorter responses when long ones are still missing")
    p.add_argument("--all-samples", action="store_true",
                   help="use every finished sample of each question (n >= k) in the unbiased pass@k estimator")
    p.add_argument("--save", help="write report + per-question predictions to this JSON")
    p.add_argument("--judge-model", default=None, help="HLE: the judge (default judge_answers.JUDGE_MODEL)")
    p.add_argument("--judge-cache", default=None,
                   help="HLE: the verdict cache (default: the one the debate search uses)")
    p.add_argument("--judge-workers", type=int, default=16, help="HLE: judge calls in flight")
    main(p.parse_args())

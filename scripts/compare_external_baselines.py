"""Our Direct CoT and Self-Refine programs against the external Direct CoT and Self-Refine baselines,
on the search questions, before a cluster search (run_pipeline_cluster.sh, step 2b): do ours match theirs?

  ours                                       external (baselines/, same questions, 3 runs)
  direct_high       one high-effort solver   direct: the dataset's own prompt (SuperGPQA's zero-shot
                                             prompt; MATH's request for a \\boxed{} answer; HLE's
                                             response format), thinking on
  self_refine_high  solver, then up to 2     Self-Refine: answer, then up to 2 feedback > refine
                    critic > solver rounds;  rounds; stops when the feedback says "it is correct"
                    stops after a critic
                    that keeps the answer

Ours run as many times per question as the external program has runs (its score file's; replicates
0, 1, ...), under the search's own settings and round cache. Generation 0 of the search runs every seed
at replicates 0 and 1, so it replays the ones run here and runs the rest (HLE, one external run: the
check runs replicate 0 and generation 0 runs replicate 1). A low-effort solver (the direct program,
replicate 0) is also run, for the ratio of high-effort to low-effort tokens.

Checks (all must hold):
  - accuracy:    for each program, ours minus external (paired over questions; both over the same
                 number of runs, 3 or (HLE) 1) is not clearly below zero: difference + 2 standard
                 errors >= 0
  - out of room: the high-effort solver's share of replies with no visible text is at most 2 points
                 above the external direct baseline's share of replies cut off
  - no letter:   at most 1% of each program's debates end with no letter (MATH and HLE: no answer)
HLE (--answers open): ours and the external answers are both graded by the judge model, with the
one verdict cache the setup names (--judge-cache), so an answer both give is judged once.
With about 200 questions the accuracy check only catches a gap of about 4 points or more; the
numbers are shown in full. Token use is not a check (removed 2026-10-07: on gpt-oss our Self-Refine
stops after the first critic far more often than the external one, so it used 30% fewer tokens at no
loss of accuracy); ours and the external programs' tokens per question are printed and reported.

    python scripts/compare_external_baselines.py --questions outputs/pipeline_cluster_qwen9b/search_questions_train100.json \\
        --out outputs/pipeline_cluster_qwen9b/run2/external_baselines \\
        --external-direct baselines/results/direct_qwen35_9b_think_clusters_600_train_search_train100_rec_k3.json \\
        --external-direct-raw baselines/results/direct_qwen35_9b_think_clusters_600_train_search_train100.jsonl \\
        --external-selfrefine baselines/results/selfrefine_qwen35_9b_think_clusters_600_train_search_train100_rec_k3.json \\
        --model Qwen/Qwen3.5-9B --base-urls http://localhost:7472/v1 \\
        --live-cache outputs/pipeline_cluster_qwen9b/rounds_qwen35_9b.jsonl \\
        --plain-instruction --last-round-vote --count-read-summaries --high-cost 3 --turn-cap 15

Writes <out>.json (the verdict and the numbers) and <out>.md. Exit code 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_debate_mcq as B  # noqa: E402
import debate_mcq as D  # noqa: E402
import program_space as P  # noqa: E402

ROOM_MARGIN = 0.02               # out-of-room share above the external baseline's cut-off share
NO_LETTER_LIMIT = 0.01           # debates that end with no letter at all
PAIRS = (("direct_high", "direct", "Direct CoT"), ("self_refine_high", "selfrefine", "Self-Refine"))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def paired(a: list[float], b: list[float]) -> tuple[float, float]:
    d = [x - y for x, y in zip(a, b)]
    m = mean(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) if len(d) > 1 else 0.0
    return m, se


def first_reply(runner, row: dict, prog: dict, rep: int) -> str | None:
    """The recorded text of the program's first speaker (its blind first round), or None."""
    _, _, hit, _ = runner.lookup(row["id"], [], prog["plan"][0], rep, B.question_prompts(row))
    return hit[0][1] if hit else None


def run_program_reps(runner, rows: dict, qids: list[str], prog: dict, reps, workers: int, name: str) -> dict:
    """{question: [result of each replicate]} for the questions every replicate ran."""
    got = {rep: P.run_many(prog, runner, rows, qids, rep, P.RUN_CAP, workers=workers,
                           desc=f"against external, {name}, replicate {rep}") for rep in reps}
    return {q: [got[rep][q] for rep in reps] for q in qids if all(got[rep].get(q) for rep in reps)}


def describe(runner, rows: dict, prog: dict, results: dict, reps) -> dict:
    """Per question: right per replicate, counted tokens per replicate, no letter at the end, and for
    the first speaker's reply: no visible text (the thinking used the whole room)."""
    per_q = {}
    for q, outs in results.items():
        row = rows[q]
        firsts = [first_reply(runner, row, prog, rep) or "" for rep in reps]
        per_q[q] = {"right": [int(bool(o["correct"])) for o in outs],
                    "tokens": [o["tokens"] or 0 for o in outs],
                    "turns": [o["n_calls"] for o in outs],
                    "no_letter": [o["letter"] is None for o in outs],
                    "no_reply": [D._split_summary(t)[0].startswith("(no reply") for t in firsts]}
    return per_q


def external(k3_path: Path, qids: list[str]) -> dict:
    per_q = json.loads(k3_path.read_text())["per_question"]
    lacking = [q for q in qids if q not in per_q]
    if lacking:
        raise SystemExit(f"{k3_path}: {len(lacking)} of the {len(qids)} questions are missing (e.g. {lacking[0]})")
    if not all("tokens" in per_q[q] for q in qids):
        raise SystemExit(f"{k3_path}: no tokens per run (score it again with baselines/score.py --save)")
    return {q: {"marks": list(per_q[q]["marks"]), "tokens": list(per_q[q]["tokens"])} for q in qids}


def cut_off_share(raw_path: Path, qids: list[str], k: int) -> float:
    """The share of the external direct baseline's samples cut off at their token limit (before recovery)."""
    want, cut, n = set(qids), 0, 0
    for line in raw_path.open():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("id") in want and r.get("error") is None and r.get("sample_idx", 99) < k:
            n += 1
            cut += r.get("finish_reason") == "length"
    return cut / n if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, required=True, help="the search questions (a dataset file)")
    ap.add_argument("--out", type=Path, required=True, help="writes <out>.json and <out>.md")
    ap.add_argument("--external-direct", type=Path, required=True,
                    help="the external direct baseline's score file on these questions (score.py --save, k3)")
    ap.add_argument("--external-direct-raw", type=Path, required=True,
                    help="its generated samples (generate.py output, before recovery): their finish reasons")
    ap.add_argument("--external-selfrefine", type=Path, required=True,
                    help="the external Self-Refine baseline's score file on these questions (k3)")
    live = ap.add_argument_group("debate model")
    live.add_argument("--base-urls", default=P.DEFAULT_BASE_URLS)
    live.add_argument("--model", default=P.DEFAULT_MODEL)
    live.add_argument("--api-key", default="EMPTY")
    live.add_argument("--temperature", type=float, default=0.7)
    live.add_argument("--workers", type=int, default=64)
    live.add_argument("--live-cache", type=Path, required=True, help="the search's round cache")
    live.add_argument("--max-total-calls", type=int, default=100_000)
    live.add_argument("--ignore-cache-lock", action="store_true")
    P.add_executor_args(ap)
    args = ap.parse_args()

    settings = P.configure_from_args(args)
    settings["model"] = args.model
    what = "answer" if D.OPEN else "letter"          # what a debate that ends without one lacks (MATH, HLE: answer)
    rows = {r["id"]: r for r in json.loads(args.questions.read_text())}
    P.check_rows(rows.values())
    qids = list(rows)
    ext = {"direct": external(args.external_direct, qids), "selfrefine": external(args.external_selfrefine, qids)}
    # the external runs per question: as many as the score files hold (score.py --k)
    ext_runs = {b: min(len(e[q]["marks"]) for q in qids) for b, e in ext.items()}
    ext_cut = cut_off_share(args.external_direct_raw, qids, ext_runs["direct"])

    runner = P.make_runner(rows, args.live_cache, args.base_urls, args.model, args.temperature,
                           max_total_calls=args.max_total_calls, api_key=args.api_key,
                           lock=not args.ignore_cache_lock)
    per: dict[str, dict] = {}
    try:
        runner.reset_budget(None)
        # ours run as many times as the external program: replicates 0 .. runs - 1
        for name, reps in [(ours, tuple(range(ext_runs[theirs]))) for ours, theirs, _ in PAIRS] + [("direct", (0,))]:
            prog = P.normalize_program(P.PROTOCOLS[name])
            runner.stage(f"against external: {name}")
            per[name] = describe(runner, rows, prog, run_program_reps(runner, rows, qids, prog, reps,
                                                                      args.workers, name), reps)
    finally:
        runner.reset_budget(0)
        runner.close()

    rows_out, checks = [], []
    for ours, theirs, label in PAIRS:
        d = per[ours]
        done = sorted(d)
        acc = mean([mean(d[q]["right"]) for q in done])
        e_acc = mean([mean(ext[theirs][q]["marks"]) for q in done])
        diff, se = paired([mean(d[q]["right"]) for q in done], [mean(ext[theirs][q]["marks"]) for q in done])
        tok = mean([mean(d[q]["tokens"]) for q in done])
        e_tok = mean([mean(ext[theirs][q]["tokens"]) for q in done])
        no_letter = mean([mean(d[q]["no_letter"]) for q in done])
        no_reply = mean([mean(d[q]["no_reply"]) for q in done])
        rows_out.append({"label": label, "ours": ours, "questions": len(done), "accuracy": acc,
                         "accuracy_run1": mean([d[q]["right"][0] for q in done]), "external_accuracy": e_acc,
                         "external_accuracy_run1": mean([ext[theirs][q]["marks"][0] for q in done]),
                         "diff": diff, "se": se, "tokens": tok, "external_tokens": e_tok,
                         "external_runs": ext_runs[theirs],
                         "turns": mean([mean(d[q]["turns"]) for q in done]), "no_letter": no_letter,
                         "no_reply": no_reply})
        checks += [
            {"name": f"{label}: every question ran", "ok": len(done) == len(qids),
             "says": f"{len(done)} of {len(qids)} questions ran at every replicate"},
            {"name": f"{label}: accuracy", "ok": diff + 2 * se >= 0,
             "says": f"ours {acc:.1%}, external {e_acc:.1%} ({ext_runs[theirs]} run"
                     f"{'s' if ext_runs[theirs] > 1 else ''} each): {diff:+.1%} +- {se:.1%}"},
            {"name": f"{label}: no {what}", "ok": no_letter <= NO_LETTER_LIMIT,
             "says": f"{no_letter:.1%} of debates end with no {what} (limit {NO_LETTER_LIMIT:.0%})"}]
    high = rows_out[0]
    checks.append({"name": "Direct CoT: out of room", "ok": high["no_reply"] <= ext_cut + ROOM_MARGIN,
                   "says": f"{high['no_reply']:.1%} of high-effort solver replies have no visible text; external "
                           f"direct cut off {ext_cut:.1%} (limit {ext_cut + ROOM_MARGIN:.1%})"})
    low = per["direct"]
    low_tokens = mean([low[q]["tokens"][0] for q in low])
    low_acc = mean([low[q]["right"][0] for q in low])
    high_first_tokens = mean([per["direct_high"][q]["tokens"][0] for q in per["direct_high"]])
    ratio = high_first_tokens / low_tokens if low_tokens else float("nan")
    checks.append({"name": "low-effort solver ran", "ok": len(low) == len(qids),
                   "says": f"{len(low)} of {len(qids)} questions"})
    verdict = "pass" if all(c["ok"] for c in checks) else "fail"
    result = {"verdict": verdict, "checks": checks, "settings": settings, "n_questions": len(qids),
              "programs": rows_out, "external_direct_cut_off_share": ext_cut,
              "low_effort_solver": {"accuracy": low_acc, "tokens": low_tokens},
              "high_to_low_tokens": ratio, "per_question": per}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=1))

    md = [f"# Ours against the external baselines: {verdict.upper()}", "",
          f"{args.model}, {len(qids)} search questions. Ours and external: "
          f"{' and '.join(sorted({str(n) for n in ext_runs.values()}))} run{'s' if max(ext_runs.values()) > 1 else ''} "
          f"per question each. "
          f"Settings: `{json.dumps(settings)}`.", "",
          "## Checks", "", "| check | result | pass |", "|---|---|---|"]
    md += [f"| {c['name']} | {c['says']} | {'yes' if c['ok'] else 'NO'} |" for c in checks]
    md += ["", "## Ours against external", "",
           f"| program | ours avg@{max(ext_runs.values())} | ours run 1 | external avg@{max(ext_runs.values())} | external run 1 | ours - external | "
           "ours tokens/q | external tokens/q | ours turns/q |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows_out:
        md.append(f"| {r['label']} ({r['ours']}) | {r['accuracy']:.1%} | {r['accuracy_run1']:.1%} | "
                  f"{r['external_accuracy']:.1%} | {r['external_accuracy_run1']:.1%} | {r['diff']:+.1%} +- {r['se']:.1%} | "
                  f"{r['tokens']:,.0f} | {r['external_tokens']:,.0f} | {r['turns']:.1f} |")
    md += ["", "## Token use (not a check)", "",
           "| program | ours tokens/q | external tokens/q | ours / external | ours turns/q |",
           "|---|---:|---:|---:|---:|"]
    md += [f"| {r['label']} ({r['ours']}) | {r['tokens']:,.0f} | {r['external_tokens']:,.0f} | "
           f"{r['tokens'] / r['external_tokens']:.2f} | {r['turns']:.1f} |" for r in rows_out]
    md += ["",
           f"- Low-effort solver (direct, one run): {low_acc:.1%} at {low_tokens:,.0f} counted tokens per question. "
           f"High-effort to low-effort tokens: {ratio:.2f}; a high-effort speaker counts {settings.get('high_cost')} turns.",
           f"- Counted tokens leave out a summary that no later speaker read"
           f"{'' if settings.get('count_read_summaries') else ' (off in this run: every token is counted)'}.",
           "- The accuracy check catches only a clear gap (about 4 points on 200 questions); read the "
           "differences above.", ""]
    args.out.with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"-> {args.out.with_suffix('.md')}, {args.out.with_suffix('.json')}")
    sys.exit(0 if verdict == "pass" else 1)


if __name__ == "__main__":
    main()

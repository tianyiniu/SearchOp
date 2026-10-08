"""Seed stage of the global-pipeline program search (one search over the train split, three cost levels).

    8   literature      the same 8 as the cluster-pipeline seed stage (program_seeds_cluster.literature_seeds:
                        mad, early_exit_agree, expert_first and the five high-effort
                        programs); the judge seeds only if --judge-persona is on (global-pipeline runs
                        without it)
    3   model-written   one per cost level (evolve_pipeline_global.LEVELS), written in one batch
                        by the guide model (program_guide.GUIDE_MODEL). It is shown the
                        grammar, the literature programs, the levels and --seed-examples
                        example questions drawn at random from the train split (question
                        and options, as the debaters see them; no answers, no difficulty
                        labels, no scores). No dev question is ever shown. The written
                        programs are cached beside the output (<out>.llm.json) after every
                        reply and reused; a run stopped by a failed call goes on, on its
                        next start, with the levels still missing (no paid reply is lost).

A level the guide model leaves short (after its retries) is left out: there are no random
stand-ins. The seed set is fixed once written.

    python scripts/program_seeds_global.py --splits <run>/splits.json --out <run>/seeds.json \\
        --dataset datasets/supergpqa_600_train.json [executor options]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evolve_pipeline_global as V4  # noqa: E402
import program_guide as G  # noqa: E402
import program_seeds as S  # noqa: E402
import program_seeds_cluster as S3  # noqa: E402
import program_space as P  # noqa: E402
from split_train_dev import load_split  # noqa: E402

ROOT = P.ROOT


def example_questions(train: list[str], n: int, seed: int) -> list[str]:
    """`n` train questions drawn at random (a fixed draw for a given seed), in draw order."""
    return random.Random(seed).sample(sorted(train), min(n, len(train)))


def write_json(path: Path, obj) -> None:
    """Write `path` whole or not at all (a stop mid-write leaves the old file)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, required=True, help="the train / dev split (split_train_dev.py)")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_600_train.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="the draw of the example questions")
    ap.add_argument("--no-guide", action="store_true", help="the literature seeds only")
    ap.add_argument("--guide-effort", default=G.GUIDE_EFFORT)
    ap.add_argument("--seed-examples", type=int, default=10, help="example train questions shown to the guide model")
    ap.add_argument("--seed-example-chars", type=int, default=2500,
                    help="an example question longer than this keeps its start and its end")
    live = ap.add_argument_group("debate model (only its window is read, to label the seeds)")
    live.add_argument("--base-urls", default=P.DEFAULT_BASE_URLS)
    live.add_argument("--model", default=P.DEFAULT_MODEL)
    live.add_argument("--api-key", default="EMPTY")
    P.add_executor_args(ap)
    args = ap.parse_args()

    settings = P.configure_from_args(args)
    settings["model"] = args.model
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; the seed set is fixed once written (move it to redo)")
    splits = load_split(args.splits)
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}
    lacking = [q for q in splits["train"] if q not in rows]
    if lacking:
        raise SystemExit(f"{len(lacking)} train questions of {args.splits} are not in {args.dataset}")
    P.check_rows([rows[q] for q in splits["train"]])
    args.out.parent.mkdir(parents=True, exist_ok=True)

    seeds: list[dict] = []
    for name, (source, prog) in S3.fixed_seeds().items():
        P.validate_program(prog)
        seeds.append({"name": name, "source": source, "program": P.normalize_program(prog)})
    shown: list[str] = []
    if not args.no_guide:
        llm_path = args.out.with_suffix(".llm.json")
        cached = json.loads(llm_path.read_text()) if llm_path.exists() else None
        written: list[dict] = []
        if cached is not None:
            written, shown = cached["programs"], cached["examples"]
            # reuse only what was written for this split and these settings
            if cached.get("settings") != settings:
                raise SystemExit(f"{llm_path} was written under {cached.get('settings')}, not {settings}; "
                                 f"move it to have the seeds written anew")
            if set(shown) - set(splits["train"]):
                raise SystemExit(f"{llm_path} was written from {len(set(shown) - set(splits['train']))} example "
                                 f"question(s) that are not in this train split; move it to have the seeds "
                                 f"written anew")
            print(f"reusing {len(written)} model-written programs from {llm_path}")
        else:
            shown = example_questions(splits["train"], args.seed_examples, args.seed)
        if cached is None or not cached.get("done", True):
            assert not set(shown) & set(splits["dev"])
            text = P.questions_text(shown, rows, f"  example questions ({len(shown)} of the "
                                                 f"{len(splits['train'])} train questions; answers not shown):",
                                    args.seed_example_chars)
            if cached is not None:
                print(f"{llm_path} is unfinished (a call failed): asking for the levels still missing")

            def keep(progs: list[dict], done: bool = False) -> None:
                write_json(llm_path, {"settings": settings, "examples": shown, "programs": progs, "done": done})

            written = G.write_level_seeds(G.make_client(), {s["name"]: s["program"] for s in seeds}, text,
                                          V4.level_text(P.MAX_TURNS), effort=args.guide_effort,
                                          n_examples=len(shown), written=written, on_reply=keep)
            keep(written, done=True)
            print(f"{len(written)} model-written programs ({G.GUIDE_MODEL}) -> {llm_path}")
        used = {s["name"] for s in seeds}
        taken = {P.canon(s["program"]) for s in seeds}
        for w in written:
            P.validate_program(w["program"])
            if P.canon(w["program"]) in taken or w["name"] in used:
                continue
            used.add(w["name"])
            taken.add(P.canon(w["program"]))
            seeds.append({"name": w["name"], "source": "llm", "level": w["level"], "program": w["program"],
                          "strategy": w.get("strategy", "")})
        short = [name for name, _ in V4.LEVELS if name not in {s.get("level") for s in seeds}]
        if short:
            print(f"note: no model-written seed for level(s) {short}")

    for s in seeds:
        s["summary"] = S.summarize(s["program"])
    out = {"settings": settings, "splits": str(args.splits), "seed": args.seed,
           "guide_model": None if args.no_guide else G.GUIDE_MODEL, "examples": shown, "seeds": seeds}
    report = [f"# Seed programs ({len(seeds)})", "", f"Executor: {settings}", "",
              "| name | source | written for | plan | rules | extra rounds | stop reads |",
              "|---|---|---|---|---|---|---|"]
    for s in seeds:
        sm = s["summary"]
        report.append(f"| {s['name']} | {s['source']} | {s.get('level', '-')} | {' > '.join(sm['plan'])} | "
                      f"{sm['n_rules']} | {', '.join(sm['moves']) or '-'} | {', '.join(sm['stop_reads'])} |")
    report += ["", "## Programs", ""]
    for s in seeds:
        report.append(f"### {s['name']} ({s['source']}" + (f", {s['level']}" if s.get("level") else "") + ")")
        if s.get("strategy"):
            report.append(s["strategy"])
        report += ["```json", json.dumps(s["program"], indent=1), "```", ""]
    args.out.with_suffix(".md").write_text("\n".join(report))
    write_json(args.out, out)               # last: the pipeline takes an existing seeds.json as done
    print(f"{len(seeds)} seeds -> {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()

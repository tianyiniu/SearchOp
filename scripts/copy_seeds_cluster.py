"""Copy an earlier run's seeds into a new run, with each literature seed (source "protocol") replaced
by its current definition in program_space.PROTOCOLS, so the new run searches from the programs the
code now defines (2026-10-06: self_refine_high stops after the critic when the critic keeps the
answer). The model-written seeds are kept as they are. Every seed must run as written under the new turn cap (program_space.never_runs_as_written).

    python scripts/copy_seeds_cluster.py --source outputs/pipeline_cluster_qwen9b/run1/seeds.json \\
        --out outputs/pipeline_cluster_qwen9b/run2/seeds.json --context-window 32768 \\
        --plain-instruction --last-round-vote --count-read-summaries --high-cost 3 --turn-cap 15

An existing --out is kept as it is (the run may have started from it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import program_space as P  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="the earlier run's seeds.json")
    ap.add_argument("--out", type=Path, required=True, help="the new run's seeds.json")
    ap.add_argument("--model", default=P.DEFAULT_MODEL)
    ap.add_argument("--base-urls", default=P.DEFAULT_BASE_URLS)
    P.add_executor_args(ap)
    args = ap.parse_args()
    if args.out.exists():
        print(f"{args.out} exists; kept as it is")
        return
    settings = P.configure_from_args(args)
    settings["model"] = args.model
    seeds = json.loads(args.source.read_text())
    changed, bad = [], []
    for s in seeds["seeds"]:
        if s.get("source") == "protocol":
            if s["name"] not in P.PROTOCOLS:
                raise SystemExit(f"{args.source}: literature seed {s['name']!r} is not in program_space.PROTOCOLS")
            now = P.normalize_program(P.PROTOCOLS[s["name"]])
            if P.canon(now) != P.canon(P.normalize_program(s["program"])):
                changed.append(s["name"])
            s["program"] = now
        prog = P.normalize_program(s["program"])
        P.validate_program(prog)
        if (why := P.never_runs_as_written(prog)) is not None:
            bad.append(f"{s['name']}: {why}")
        if P.reviewer_first(prog):
            bad.append(f"{s['name']}: a critic, verifier or synthesizer speaks in its first round")
    if bad:
        raise SystemExit("seeds the search may not start from:\n  " + "\n  ".join(bad))
    seeds["copied_from"] = {"seeds": str(args.source), "settings": seeds.get("settings"),
                            "literature_seeds_redefined": changed}
    seeds["settings"] = settings
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(seeds, indent=1))
    print(f"{len(seeds['seeds'])} seeds -> {args.out}; literature seeds replaced by their current definition: "
          f"{changed or 'none'}")


if __name__ == "__main__":
    main()

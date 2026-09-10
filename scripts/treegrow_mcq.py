"""Method 1: grow a debate PER QUESTION, one round at a time, by tree search.

Bottom-up per-question construction. A node is a real, executed debate transcript;
an edge is "what round happens next"; a leaf is a finished debate. Because rounds
execute as the tree grows, the choice of round 3 can depend on what was actually
said in rounds 1-2 -- which is what makes this per-question, unlike every global
recipe search in this repo.

Two modes:

  --mode train        Beam search steered by the answer key. Every expansion is
                      recorded as a LESSON (state -> move -> worked/failed); the
                      winning root-to-leaf path is the question's own recipe, the
                      dead siblings are its negative examples.

                      A hit is VERIFIED before it counts: the path is re-run from
                      scratch --verify-reps times (fresh samples of every round,
                      cached as replicates 1..k) and confirmed only if at least
                      --verify-min reruns are also correct. A lucky rollout is
                      demoted to "unsolved" and keeps growing, so the search
                      routes around flukes instead of stopping on them.

  --mode controller   Deployable inference: no key. After each round one extra
                      model call reads the transcript plus ~12 train-time lessons
                      from similar situations and picks the next move.

Moves (contrarian-free by design):
  critic     one critic (the E7 rewritten prompt) reads all, hunts for a flaw
  verifier   tests each committed letter against the question's exact wording
  fresh      two blind independent solvers
  expert     one blind field-named expert (field injected per question)
  eliminate  eliminator strikes options (answers nothing), then a solver picks
  stop       finalize; answer = last committed letter

    python3 scripts/treegrow_mcq.py --mode train
    python3 scripts/treegrow_mcq.py --mode controller
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import debate_mcq as D
import schema_fitness as SF
from schema_fitness import BudgetExhausted, Pool, RoundRunner, load_base_letters, log

ROOT_SPEC = {"personas": ["solver"]}

# Each move appends these rounds. Order matters: it is the order personas run.
ACTIONS: dict[str, list[dict]] = {
    "critic":    [{"personas": ["critic"], "sees": "all"}],
    "verifier":  [{"personas": ["verifier"], "sees": "all"}],
    "fresh":     [{"personas": ["independent", "independent"]}],   # forced blind
    "expert":    [{"personas": ["expert"], "sees": "none"}],
    "eliminate": [{"personas": ["eliminator"], "sees": "all"},
                  {"personas": ["solver"], "sees": "all"}],
}
CHOICES = tuple(ACTIONS) + ("stop",)


def question_prompts(row: dict, plain_critic: bool = False) -> dict:
    """Per-question persona-prompt overrides: the field-named expert, plus the
    E7 rewritten critic unless --plain-critic."""
    p = {"expert": D.EXPERT_TMPL.format(
        field=row.get("field") or row.get("discipline") or "the relevant field")}
    if not plain_critic:
        p["critic"] = SF.REWRITTEN_CRITIC
    return p


def standing(rounds: list, n: int) -> str | None:
    """The debate's current answer: the last committed letter, if any."""
    letters = SF.committed_letters(rounds, n)
    return letters[-1] if letters else None


def pattern(rounds: list, n: int) -> str:
    """Compressed state view for lessons: each speaker and its letter."""
    parts = []
    for rnd in rounds:
        for p, r in rnd:
            parts.append(f"{p} -" if p in D.NON_ANSWERING
                         else f"{p} {D.extract_letter(r, n) or '?'}")
    return "; ".join(parts)


def state_type(rounds: list, n: int) -> str:
    """Coarse situation label used to match lessons to live states."""
    distinct = set(SF.committed_letters(rounds, n))
    if len(distinct) <= 1:
        return "parked"
    return "moved" if len(distinct) == 2 else "scattered"


@dataclass
class Node:
    specs: list            # round specs executed, in order
    rounds: list           # their transcripts
    actions: list          # move names that built this node
    parent: "Node | None" = None
    depth: int = 0
    solved: bool = False           # confirmed by reruns (or raw hit when verification is off)
    raw_hit: bool = False          # this rollout's standing letter == gold
    rerun_hits: int = 0            # how many verification reruns were also correct
    lesson: dict | None = field(default=None, repr=False)


def expand(runner: RoundRunner, qid: str, parent: Node, aname: str,
           prompts: dict, rep: int) -> Node:
    """Run one move on top of `parent`. The parent's rounds are shared, not
    re-run -- only the new round(s) cost anything."""
    specs, rounds = list(parent.specs), list(parent.rounds)
    for spec in ACTIONS[aname]:
        rounds = rounds + [runner.run_round(qid, rounds, specs, spec, rep=rep,
                                            prompts=prompts)]
        specs = specs + [spec]
    return Node(specs=specs, rounds=rounds, actions=parent.actions + [aname],
                parent=parent, depth=parent.depth + 1)


def select_beam(children: list[Node], width: int, n: int) -> list[Node]:
    """Among unsolved children keep `width`, preferring distinct standing letters
    (diversity is the only signal when everything is wrong), parseable letters
    before None, original order otherwise."""
    ranked = sorted(children, key=lambda c: standing(c.rounds, n) is None)
    chosen, seen = [], set()
    for c in ranked:
        l = standing(c.rounds, n)
        if len(chosen) < width and l not in seen:
            chosen.append(c)
            seen.add(l)
    for c in ranked:
        if len(chosen) >= width:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


# --- train mode -------------------------------------------------------------

def replay_path(runner: RoundRunner, qid: str, specs: list, rep: int,
                prompts: dict) -> list:
    """Execute a path from scratch under a new replicate: fresh samples of every
    round, root included. Tests the RECIPE, not the original transcript."""
    rounds, done = [], []
    for spec in specs:
        rounds.append(runner.run_round(qid, rounds, done, spec, rep=rep, prompts=prompts))
        done.append(spec)
    return rounds


def score(runner: RoundRunner, qid: str, node: Node, gold: str, n: int,
          prompts: dict, args) -> None:
    """Set node.raw_hit / rerun_hits / solved. A raw hit is re-run --verify-reps
    times; it is a confirmed solve only if >= --verify-min reruns also hit."""
    node.raw_hit = standing(node.rounds, n) == gold
    if not node.raw_hit:
        return
    if args.verify_reps <= 0:
        node.solved = True
        return
    node.rerun_hits = sum(
        standing(replay_path(runner, qid, node.specs, r, prompts), n) == gold
        for r in range(1, args.verify_reps + 1))
    node.solved = node.rerun_hits >= args.verify_min


def search_question(runner: RoundRunner, row: dict, args) -> tuple[dict, list[dict]]:
    """Beam search one question. Returns (question record, lesson rows)."""
    qid, gold, n = row["id"], row["answer_letter"], len(row["options"])
    prompts = question_prompts(row, args.plain_critic)
    root = Node(specs=[ROOT_SPEC],
                rounds=[runner.run_round(qid, [], [], ROOT_SPEC, rep=args.rep,
                                         prompts=prompts)],
                actions=[])
    score(runner, qid, root, gold, n, prompts, args)
    all_nodes, lessons = [root], []
    correct = [root] if root.solved else []
    beam = [] if root.solved else [root]

    def grow(task):
        child = expand(runner, qid, task[0], task[1], prompts, args.rep)
        score(runner, qid, child, gold, n, prompts, args)
        return child

    for depth in range(1, args.depth + 1):
        if not beam:
            break
        tasks = [(node, aname) for node in beam for aname in ACTIONS]
        if args.inner_workers > 1:
            with ThreadPoolExecutor(max_workers=args.inner_workers) as pool:
                children = list(pool.map(grow, tasks))
        else:
            children = [grow(t) for t in tasks]
        for (node, aname), child in zip(tasks, children):
            child.lesson = {"qid": qid,
                            "field": row.get("field") or row.get("discipline") or "",
                            "n_options": n, "depth": depth,
                            "pattern": pattern(node.rounds, n),
                            "state_type": state_type(node.rounds, n),
                            "path_actions": node.actions, "action": aname,
                            "child_letter": standing(child.rounds, n),
                            "raw_hit": child.raw_hit, "rerun_hits": child.rerun_hits,
                            "child_solved": child.solved,       # confirmed
                            "subtree_solved": child.solved}
            lessons.append(child.lesson)
            all_nodes.append(child)
        correct += [c for c in children if c.solved]        # confirmed -> leaf, stop growing
        beam = select_beam([c for c in children if not c.solved], args.beam, n)

    for node in all_nodes:                                  # backfill subtree_solved
        if node.solved:
            p = node
            while p is not None:
                if p.lesson is not None:
                    p.lesson["subtree_solved"] = True
                p = p.parent

    first = min(correct, key=lambda c: c.depth) if correct else None
    raw = [c for c in all_nodes if c.raw_hit]
    rec = {"qid": qid, "gold": gold, "solved": bool(correct),
           "raw_solved": bool(raw),
           "root_letter": standing(root.rounds, n),
           "solve_depth": first.depth if first else None,
           "winning_path": first.actions if first else None,
           "winning_rerun_hits": first.rerun_hits if first else None,
           "n_raw_hits": len(raw), "n_confirmed": len(correct),
           "n_flukes": len(raw) - len(correct), "n_nodes": len(all_nodes)}
    return rec, lessons


# --- controller mode --------------------------------------------------------

CONTROLLER_SYSTEM = """You manage a debate between AI assistants answering a hard multiple-choice question. After each round you pick the next move. The moves:

critic    - one critic reads everything so far and hunts for a flaw in the current answer.
verifier  - one verifier tests each committed letter against the exact wording of the question.
fresh     - two new solvers answer independently, seeing nothing of the debate.
expert    - one field expert answers independently, seeing nothing of the debate.
eliminate - an eliminator rules out impossible options, then a solver picks from the survivors.
stop      - end the debate; the most recent committed answer becomes final.

You will see examples of past debates where a move succeeded or failed from a similar situation, then the live debate. Reply with EXACTLY ONE WORD: the move name."""


_CHOICE_WORDS = {**{c: c for c in CHOICES},
                 "eliminator": "eliminate"}      # the persona's name, a likely reply


def parse_choice(text: str | None) -> str | None:
    t = (text or "").lower()
    hits = [(t.find(w), a) for w, a in _CHOICE_WORDS.items() if t.find(w) != -1]
    return min(hits)[1] if hits else None


def render_lesson(r: dict) -> str:
    got = r["child_letter"] or "no parseable answer"
    fate = "eventually SUCCEEDED" if r["subtree_solved"] else "FAILED"
    return (f"Debate so far: {r['pattern']}. Move tried: {r['action']} -> "
            f"answered {got}; this line of attack {fate}.")


def load_lessons(path: Path) -> dict:
    bank = defaultdict(list)
    for line in Path(path).open():
        line = line.strip()
        if line:
            r = json.loads(line)
            bank[(r["depth"], r["state_type"])].append(r)
    return bank


def pick_lessons(bank: dict, depth: int, stype: str, k: int,
                 rng: random.Random) -> list[dict]:
    """~k lessons for this situation: exact (depth, type) bucket first, then same
    type at any depth, then anything; balanced between worked and failed."""
    pool = (bank.get((depth, stype))
            or [r for (d, s), v in bank.items() if s == stype for r in v]
            or [r for v in bank.values() for r in v])
    worked = [r for r in pool if r["subtree_solved"]]
    failed = [r for r in pool if not r["subtree_solved"]]
    half = k // 2
    sel = (rng.sample(worked, min(half, len(worked)))
           + rng.sample(failed, min(k - half, len(failed))))
    rng.shuffle(sel)
    return sel


def controller_user(row: dict, rounds: list, lessons: list[dict], n: int) -> str:
    les = "\n".join(f"- {render_lesson(r)}" for r in lessons) or "- (none available)"
    ctx = D._visible(rounds, "all", n)
    return (f"Past examples from similar situations:\n{les}\n\n"
            f"The live question ({row.get('field') or '?'}, {n} options):\n"
            f"{D.render_question(row['question'], list(row['options']))}\n\n{ctx}\n\n"
            f"Pick the next move. Reply with one word.")


def run_controller_question(runner: RoundRunner, ctrl_client, ctrl_model: str,
                            row: dict, bank: dict, args) -> dict:
    qid, gold, n = row["id"], row["answer_letter"], len(row["options"])
    prompts = question_prompts(row, args.plain_critic)
    rng = random.Random(args.seed * 100003
                        + int(hashlib.sha1(qid.encode()).hexdigest()[:8], 16))
    specs = [ROOT_SPEC]
    rounds = [runner.run_round(qid, [], [], ROOT_SPEC, rep=args.rep, prompts=prompts)]
    picks, ctrl_calls = [], 0
    for depth in range(1, args.depth + 1):
        lessons = pick_lessons(bank, depth, state_type(rounds, n), args.lessons, rng)
        user = controller_user(row, rounds, lessons, n)
        choice = None
        for _ in range(2):
            runner.charge(1)
            ctrl_calls += 1
            choice = parse_choice(D.chat(ctrl_client, ctrl_model, CONTROLLER_SYSTEM,
                                         user, args.controller_temperature, 512))
            if choice:
                break
        choice = choice or "critic"                       # unparseable twice -> safest move
        picks.append(choice)
        if choice == "stop":
            break
        for spec in ACTIONS[choice]:
            rounds.append(runner.run_round(qid, rounds, specs, spec, rep=args.rep,
                                           prompts=prompts))
            specs.append(spec)
    final = standing(rounds, n)
    return {"qid": qid, "gold": gold, "letter": final,
            "correct": final is not None and final == gold, "picks": picks,
            "persona_calls": sum(len(s["personas"]) for s in specs),
            "controller_calls": ctrl_calls}


# --- driver -----------------------------------------------------------------

def run_over_pool(fn, qids, workers, runner, label):
    done, results, lock, stop = [0], [], threading.Lock(), [False]

    def work(qid):
        if stop[0]:
            return None
        try:
            r = fn(qid)
        except BudgetExhausted:
            stop[0] = True
            return None
        with lock:
            done[0] += 1
            runner.stage(f"{label} {done[0]}/{len(qids)}")
        return r

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [r for r in pool.map(work, qids) if r is not None]
    if stop[0]:
        log("  call budget exhausted -- partial results below")
    return results


def main(args):
    base = load_base_letters(args.bestofn_cache)
    pool = Pool(args.dataset, base, seed=args.seed, limit=args.pool_limit)
    qids = pool.batch(args.batch) if args.batch else list(pool.order)
    runner = RoundRunner(pool.rows, args.base_urls, args.model, args.temperature,
                         args.answer_tokens, args.cache, max_calls=args.max_calls,
                         api_key=args.api_key, progress=args.progress)
    log(f"treegrow --mode {args.mode}: {len(qids)} questions | beam {args.beam} "
        f"depth {args.depth} | cache {args.cache}")

    if args.mode == "train":
        out = run_over_pool(lambda q: search_question(runner, pool.rows[q], args),
                            qids, args.workers, runner, "train")
        records = [r for r, _ in out]
        lessons = [l for _, ls in out for l in ls]
        with args.lessons_out.open("w") as fh:                 # fresh file per run
            for l in lessons:
                fh.write(json.dumps(l, ensure_ascii=False) + "\n")
        solved = [r for r in records if r["solved"]]
        raw_solved = [r for r in records if r["raw_solved"]]
        by_depth = Counter(r["solve_depth"] for r in solved)
        by_action = Counter(a for r in solved for a in (r["winning_path"] or []))
        act_rate = {a: (sum(l["raw_hit"] for l in lessons if l["action"] == a),
                        sum(l["child_solved"] for l in lessons if l["action"] == a),
                        sum(1 for l in lessons if l["action"] == a)) for a in ACTIONS}
        rerun_hist = Counter(l["rerun_hits"] for l in lessons if l["raw_hit"])
        summary = {"mode": "train", "dataset": str(args.dataset), "n": len(records),
                   "n_solved": len(solved), "n_raw_solved": len(raw_solved),
                   "search_accuracy": len(solved) / len(records) if records else 0.0,
                   "raw_search_accuracy": (len(raw_solved) / len(records)) if records else 0.0,
                   "verify_reps": args.verify_reps, "verify_min": args.verify_min,
                   "rerun_hits_among_raw_hits": {str(k): v for k, v in sorted(rerun_hist.items())},
                   "solves_by_depth": {str(k): v for k, v in sorted(by_depth.items())},
                   "winning_path_actions": dict(by_action.most_common()),
                   "action_solve_rate": {a: {"raw_hits": rh, "confirmed": k, "tried": t,
                                             "raw_rate": rh / t if t else 0.0,
                                             "confirmed_rate": k / t if t else 0.0}
                                         for a, (rh, k, t) in act_rate.items()},
                   "beam": args.beam, "depth": args.depth, "rep": args.rep,
                   "calls_spent": runner.calls, "rounds_run": runner.rounds_run,
                   "errors": runner.errors, "n_lessons": len(lessons),
                   "records": records}
        runner.close()
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nconfirmed search accuracy: {summary['search_accuracy']:.1%} "
              f"({len(solved)}/{len(records)})   raw (any hit): "
              f"{summary['raw_search_accuracy']:.1%} ({len(raw_solved)}/{len(records)})")
        print(f"rerun hits among raw hits: {summary['rerun_hits_among_raw_hits']}")
        print(f"solves by depth: {summary['solves_by_depth']}")
        print(f"winning-path moves: {summary['winning_path_actions']}")
        print(f"lessons -> {args.lessons_out} ({len(lessons)} rows)")

    else:                                                      # controller
        if not args.lessons_out.exists():
            raise SystemExit(f"{args.lessons_out} not found -- run --mode train first")
        bank = load_lessons(args.lessons_out)
        from openai import OpenAI
        ctrl_client = OpenAI(
            base_url=args.controller_base_url or args.base_urls.split(",")[0].strip(),
            api_key=args.controller_api_key or args.api_key, timeout=900.0, max_retries=2)
        ctrl_model = args.controller_model or args.model
        records = run_over_pool(
            lambda q: run_controller_question(runner, ctrl_client, ctrl_model,
                                              pool.rows[q], bank, args),
            qids, args.workers, runner, "controller")
        k = sum(r["correct"] for r in records)
        summary = {"mode": "controller", "dataset": str(args.dataset),
                   "controller_model": ctrl_model, "n": len(records), "n_correct": k,
                   "accuracy": k / len(records) if records else 0.0,
                   "pick_distribution": dict(Counter(p for r in records
                                                     for p in r["picks"]).most_common()),
                   "avg_persona_calls": (sum(r["persona_calls"] for r in records)
                                         / len(records)) if records else 0.0,
                   "avg_controller_calls": (sum(r["controller_calls"] for r in records)
                                            / len(records)) if records else 0.0,
                   "depth": args.depth, "rep": args.rep, "calls_spent": runner.calls,
                   "errors": runner.errors, "records": records}
        runner.close()
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\ncontroller accuracy: {summary['accuracy']:.1%} ({k}/{len(records)})")
        print(f"picks: {summary['pick_distribution']}")
        print(f"avg calls/question: {summary['avg_persona_calls']:.1f} persona "
              f"+ {summary['avg_controller_calls']:.1f} controller")

    print(f"calls spent: {runner.calls}  rounds: {runner.rounds_run}  "
          f"errors: {runner.errors}")
    print(f"round cache -> {args.cache}\nsummary -> {args.summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("train", "controller"), required=True)
    ap.add_argument("--dataset", type=Path, default=Path("datasets/supergpqa_strict_train.json"))
    ap.add_argument("--bestofn-cache", type=Path, default=Path("outputs/bestofn_strict_cache.jsonl"))
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=None, help="First N of the seeded shuffle; default all.")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--plain-critic", action="store_true",
                    help="Use the original critic prompt instead of the E7 rewrite.")
    ap.add_argument("--verify-reps", type=int, default=3,
                    help="Fresh reruns of a candidate winning path (0 disables verification).")
    ap.add_argument("--verify-min", type=int, default=2,
                    help="Reruns that must also be correct for a hit to count as solved.")
    ap.add_argument("--lessons", type=int, default=12, help="Lessons shown to the controller.")
    ap.add_argument("--controller-model", default=None)
    ap.add_argument("--controller-base-url", default=None)
    ap.add_argument("--controller-api-key", default=None)
    ap.add_argument("--controller-temperature", type=float, default=0.2)
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--answer-tokens", type=int, default=3072)
    ap.add_argument("--workers", type=int, default=None,
                    help="Questions in flight. Default: 16 in train mode (each question "
                         "also runs --inner-workers threads), 64 in controller mode.")
    ap.add_argument("--inner-workers", type=int, default=8,
                    help="Parallel expansions inside one question (train mode).")
    ap.add_argument("--pool-limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=2000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-progress", dest="progress", action="store_false", default=True)
    ap.add_argument("--cache", type=Path, default=Path("outputs/treegrow_rounds_cache.jsonl"))
    ap.add_argument("--lessons-out", type=Path, default=Path("outputs/treegrow_lessons.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()
    if args.summary_out is None:
        args.summary_out = Path(f"outputs/treegrow_{args.mode}_summary.json")
    if args.workers is None:
        args.workers = 16 if args.mode == "train" else 64
    main(args)

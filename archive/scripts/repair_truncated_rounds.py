"""Back-fill commitments into an existing round cache.

About one solver reply in five on Qwen3.5-27B was cut off at the token cap
before its ANSWER line, so it counted as silence. This walks a recorded cache,
finds every answering-persona reply with no letter, sends it the same commit
nudge the live executor now uses (debate_mcq.commit_followup) and appends the
model's commit line to the stored reply. Everything else is copied unchanged.

The repaired rounds are written to a NEW file under the commit-follow-up cache
key (c=1), so a run started with --commit-followup replays them for free and
pays only for rounds it has never seen. The source file is never modified.

CAVEAT, so the numbers are read correctly: a later round in a repaired
transcript was generated while the earlier reply was still truncated, so its
speaker did not see the back-filled commit. Round-1 votes and any read-off
over committed letters are exact; the later personas' behaviour is what it
was. For a clean end-to-end number, run a fresh replicate with the follow-up
on (eval_passk_programs.py --reps 2,3).

    python scripts/repair_truncated_rounds.py --model Qwen/Qwen3.5-27B \\
        --src outputs/program_live_rounds_cache_qwen27b.jsonl \\
        --dst outputs/program_live_rounds_cache_qwen27b_commit.jsonl --workers 32
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import debate_mcq as D  # noqa: E402
import adaptive_debate_mcq as B  # noqa: E402
import schema_fitness as SF  # noqa: E402


def rekey(k: str) -> str:
    d = json.loads(k)
    if d.get("c") == "1":              # already a repaired key (--renudge)
        return k
    d["c"] = "1"
    return json.dumps(d, sort_keys=False, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Recorded cache to read (untouched).")
    ap.add_argument("--dst", type=Path, required=True, help="Repaired cache to write (c=1 keys).")
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets/supergpqa_strict_train.json")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--base-urls", default="http://localhost:7472/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--digest-head", type=int, default=300)
    ap.add_argument("--digest-tail", type=int, default=900)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: repair only the first N rounds that need it.")
    ap.add_argument("--renudge", action="store_true",
                    help="--src is already a repaired (c=1) cache: strip every follow-up "
                         "that produced no letter and try it again with the current nudge. "
                         "Replies that never got a follow-up are nudged too.")
    args = ap.parse_args()
    if args.dst.exists():
        raise SystemExit(f"{args.dst} exists; choose a new --dst (the source is never modified)")
    if args.dst.resolve() == args.src.resolve():
        raise SystemExit("--dst must differ from --src")
    D.set_digest(args.digest_head, args.digest_tail)
    # source keys: plain recordings have no c; a repaired cache already has c=1
    D.set_commit_followup(bool(args.renudge))

    from openai import OpenAI
    urls = [u.strip() for u in args.base_urls.split(",") if u.strip()]
    clients = [OpenAI(base_url=u, api_key=args.api_key, timeout=900.0, max_retries=2) for u in urls]
    rows = {r["id"]: r for r in json.loads(args.dataset.read_text())}

    # pass 1: read everything, find what needs a commit
    records: list[dict] = []
    counts_free = {"parsed": 0}               # --renudge: old follow-ups fixed by parsing alone
    by_key: dict[tuple, list] = {}            # (q, k, r) -> responses, for prefix transcripts
    todo: list[tuple[int, int]] = []          # (record index, response index)
    bad = skipped_err = 0
    for line in tqdm(args.src.open(errors="replace"), desc="scan", unit="line", miniters=10_000):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if r.get("error") or r.get("q") not in rows:
            skipped_err += 1
            continue
        if not records:                       # first usable record: which pipeline made it?
            try:
                if json.loads(r["k"]).get("v"):
                    raise SystemExit(f"{args.src} is a v2 cache (keys carry v=2): its replies "
                                     "commit through their own summary call, so there is "
                                     "nothing for this script to repair")
            except (ValueError, AttributeError):
                pass
        n = len(rows[r["q"]]["options"])
        for j, (persona, text) in enumerate(r["responses"]):
            if persona in D.NON_ANSWERING or not text:
                continue
            if args.renudge and D.COMMIT_MARK in text:
                if D.extract_letter(text, n) is None:      # failed follow-up
                    head, tail = text.split(D.COMMIT_MARK, 1)
                    if (l := D.commit_letter_lenient(tail, n)) is not None:
                        r["responses"][j] = [persona, text + f"\nANSWER: {l}"]   # free
                        counts_free["parsed"] += 1
                    else:                                  # retry with the current nudge
                        r["responses"][j] = [persona, head]
                        todo.append((len(records), j))
                continue
            if D.extract_letter(text, n) is None:
                todo.append((len(records), j))
        by_key.setdefault((r["q"], r["k"], r["r"]), r["responses"])
        records.append(r)
    if args.limit is not None:
        todo = todo[: args.limit]
    n_resp = sum(len(r["responses"]) for r in records)
    print(f"{len(records)} rounds, {n_resp} replies; {len(todo)} replies without a letter "
          f"({len(todo) / max(1, n_resp):.1%}); {bad} unreadable lines, {skipped_err} errored/unknown skipped"
          + (f"; {counts_free['parsed']} old follow-ups fixed by parsing alone" if args.renudge else ""))

    # pass 2: nudge each one. The prompt is rebuilt exactly as execute_round built
    # it -- question, the digest of the prior rounds (looked up under their prefix
    # keys), the persona's instruction -- so the continuation sees what the reply
    # was written to. If a prefix round is missing, the reply's own text is all
    # the context, which the nudge ("the reasoning so far") still covers.
    counts = {"done": 0, "failed": 0, "letter": 0, "no_prefix": 0}
    lock = threading.Lock()

    def rebuild_user(r: dict, row: dict, persona: str) -> str:
        n = len(row["options"])
        specs = json.loads(r["k"])["rounds"]
        prompts = B.question_prompts(row)
        prefix = []
        for i in range(1, len(specs)):
            hit = by_key.get((r["q"], SF.path_key(specs[:i], prompts), r["r"]))
            if hit is None:
                with lock:
                    counts["no_prefix"] += 1
                prefix = []
                break
            prefix.append([tuple(pr) for pr in hit])
        this = specs[-1]
        mode = "none" if persona in D.FORCED_BLIND else this.get("sees", D.DEFAULT_SEES)
        ctx = D._visible(prefix, mode, n)
        base = D.render_question(row["question"], list(row["options"]))
        return (f"{base}\n\n{ctx}\n\nGive your response. {D.ANSWER_INSTR}" if ctx
                else f"{base}\n\n{D.ANSWER_INSTR}")

    def one(item: tuple[int, int]) -> None:
        i, j = item
        r = records[i]
        row = rows[r["q"]]
        n = len(row["options"])
        persona, text = r["responses"][j]
        prompts = B.question_prompts(row)
        book = {**D.PERSONA_PROMPTS, **prompts}
        user = rebuild_user(r, row, persona)
        client = clients[i % len(clients)]
        add = D.commit_followup(client, args.model, book.get(persona, D.PERSONA_PROMPTS["solver"]),
                                user, text, args.temperature, n_options=n)
        with lock:
            if add:
                r["responses"][j] = [persona, text + add]
                counts["done"] += 1
                if D.extract_letter(text + add, n) is not None:
                    counts["letter"] += 1
            else:
                counts["failed"] += 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(tqdm(pool.map(one, todo), total=len(todo), desc="commit follow-ups", unit="reply",
                  dynamic_ncols=True, smoothing=0.05))
    print(f"follow-ups: {counts['done']} answered ({counts['letter']} with a parseable letter), "
          f"{counts['failed']} failed, {counts['no_prefix']} rebuilt without a prior-round digest")

    # pass 3: write everything under the new key
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.dst.open("w") as fh:
        for r in tqdm(records, desc="write", unit="round", miniters=5_000):
            out = {"q": r["q"], "k": rekey(r["k"]), "r": r["r"], "responses": r["responses"],
                   "error": None}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} rounds -> {args.dst}")


if __name__ == "__main__":
    main()

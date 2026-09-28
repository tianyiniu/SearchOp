"""Offline test of the deep-think speaker and the longer summary: fake models,
a tiny v3 search with both switched on, and a check that a run without them is
untouched.

    python scripts/test_deep_think.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_program_clusters_v3 as T3  # noqa: E402  (installs the fake clients on import)
import debate_mcq as D  # noqa: E402
import evolve_program_clusters_v3 as V  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_seeds_v3 as S3  # noqa: E402
import program_space as P  # noqa: E402

T, check, run_cli, TMP = T3.T, T3.check, T3.run_cli, T3.TMP
DEEP_CALLS = {"n": 0, "with_cap": 0, "summaries_shown_thinking": 0}
_old_create = T.FakeCompletions.create


def _create(self, model, messages, temperature, max_tokens=None, extra_body=None, **kw):
    resp = _create_inner(self, model, messages, temperature, max_tokens, extra_body, **kw)
    resp.usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=7)     # token accounting
    return resp


def _create_inner(self, model, messages, temperature, max_tokens=None, extra_body=None, **kw):
    """The fake server: a deep-think request has no max_tokens and carries the
    reasoning switches; it answers with a long thinking field and a bare letter."""
    if messages[-1]["content"] == D.SUMMARY_NUDGE and "[the end of my private reasoning]" in messages[-2]["content"]:
        DEEP_CALLS["summaries_shown_thinking"] += 1
    if extra_body == D.DEEP_EXTRA:
        DEEP_CALLS["n"] += 1
        DEEP_CALLS["with_cap"] += max_tokens is not None
        resp = _old_create(self, model, messages, temperature, 1, extra_body)
        letter = D.extract_letter(resp.choices[0].message.content, 10)
        resp.choices[0].message.reasoning = "thinking it through at length. " * 400
        resp.choices[0].message.content = f"ANSWER: {letter}"
        return resp
    return _old_create(self, model, messages, temperature, max_tokens, extra_body)


T.FakeCompletions.create = _create


def cache_keys(path: Path) -> list[dict]:
    return [json.loads(json.loads(l)["k"]) for l in path.open() if l.strip()]


def test_off_is_untouched():
    print("switched off")
    settings = P.configure_executor()
    before = (dict(D.PERSONA_PROMPTS), sorted(M.ACTIONS), list(P.PLAN_ROUNDS), list(P.PROTOCOLS), P.grammar_text())
    settings = P.configure_executor()
    check(set(settings) == {"v2", "digest", "answer_tokens", "eliminator"}, "the settings name nothing new")
    after = (dict(D.PERSONA_PROMPTS), sorted(M.ACTIONS), list(P.PLAN_ROUNDS), list(P.PROTOCOLS), P.grammar_text())
    check(before == after and D.DEEP_PERSONA not in after[0] and D.DEEP_PERSONA not in after[1]
          and len(after[3]) == 8 and "deep" not in after[4].lower(),
          "no deep-think persona, move, plan round, protocol or grammar line")
    check(D.summary_signature() is None and D.SUMMARY_MAX_TOKENS == 320 and "120 words" in D.SUMMARY_NUDGE,
          "the summary request is the original one")
    check(D.turn_cost(["solver", "critic", "expert"]) == 3, "ordinary speakers cost one turn each")
    short, mid = "word " * 100, "word " * 400                    # 500 and 2,000 characters
    check(D._is_own_summary(short) and not D._is_own_summary(mid),
          "at 120 words the original rule decides: only a reply under 800 characters is its own summary")
    D.set_summary_words(500)
    check(D._is_own_summary(mid) and not D._is_own_summary("word " * 501),
          "at 500 words any reply within 500 words is its own summary")
    D.set_summary_words(120)


def test_search():
    print("a search with deep-think and 300-word summaries")
    T3.test_subsets()
    out = TMP / "deep"
    flags = ["--deep-think", "--summary-words", "300", "--digest-head", "1200", "--digest-tail", "1200"]
    common = ["--clusters", str(T3.CLUSTERS_V3), "--per-group", "3", "--live-cache", str(out / "rounds.jsonl"),
              "--workers", "8"] + flags
    run_cli(S3, common + ["--out", str(out / "seeds.json"), "--sanity-per-group", "1", "--min-turns", "1"])
    seeds = json.loads((out / "seeds.json").read_text())
    names = [s["name"] for s in seeds["seeds"]]
    lit = [s["name"] for s in seeds["seeds"] if s["source"] == "protocol"]
    check(len(names) == 24 and len(lit) == 8 and lit[0] == "deep_direct" and "direct" not in names,
          "24 seeds, 8 literature ones, deep_direct in direct's place", str(lit))
    check(sum(s["source"] == "random" for s in seeds["seeds"]) == 10, "still 10 random seeds")
    check(seeds["settings"].get("deep_think") == {"cost": D.DEEP_COST} and seeds["settings"].get("summary_words") == 300,
          "the seed file records both settings")
    run = out / "run"
    run_cli(V, common + ["--seeds", str(out / "seeds.json"), "--out", str(run), "--generations", "1"])
    lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    header, recs = lines[0], {d["key"]: d for d in lines[1:]}
    check(header["settings"].get("deep_think") == {"cost": D.DEEP_COST}
          and header["settings"].get("summary_words") == 300 and header["settings"]["digest"] == [1200, 1200],
          "the archive header records them")
    dd = next(d for d in recs.values() if d["name"] == "deep_direct")
    check(all(v[1] == D.DEEP_COST for v in dd["reps"]["0"].values()), f"one deep-think turn counts as {D.DEEP_COST} turns")
    check(DEEP_CALLS["n"] > 0 and DEEP_CALLS["with_cap"] == 0, "deep-think calls carry no reply cap",
          str(DEEP_CALLS))
    check(DEEP_CALLS["summaries_shown_thinking"] > 0, "its summary call is shown the end of the thinking")
    recs_c = [json.loads(l) for l in (out / "rounds.jsonl").open() if l.strip()]
    ok = all(len(r["usage"]) == len(r["responses"])
             and [u["persona"] for u in r["usage"]] == [pr[0] for pr in r["responses"]]
             and all(u["calls"] in (1, 2) and u["completion"] == 7 * u["calls"] and u["prompt"] == 100 * u["calls"]
                     for u in r["usage"]) for r in recs_c if not r["error"])
    check(ok, "every recording carries the tokens of each of its speakers (reply + summary calls)")
    keys = cache_keys(out / "rounds.jsonl")
    check(all(k.get("s") == 300 and k.get("d") == [1200, 1200] for k in keys),
          "every recording is keyed by the summary length and the digest window")
    used = sum(1 for d in recs.values() if D.DEEP_PERSONA in json.dumps(d["program"]))
    check(used >= 2, "other programs use the deep-think speaker too", f"{used} programs")
    kids = [d for d in recs.values() if d["gen"] > 0]
    check(len(kids) > 0 and all(max(v[1] for v in d["reps"]["0"].values()) <= 16 for d in recs.values()),
          "the 16-turn cap holds with the weighted cost")
    run_cli(V, common + ["--out", str(run), "--pick-champions", "--heldout-cap", "3",
                         "--baselines", "direct,self_refine,deep_direct"])
    champs = json.loads((run / "champions.json").read_text())
    check("deep_direct" in champs["baselines"], "deep_direct is scored as a baseline on the held-out questions")
    try:
        run_cli(V, [a for a in common if a != "--deep-think"] + ["--out", str(run), "--resume", "--generations", "1"])
        check(False, "resuming without --deep-think is refused")
    except SystemExit:
        check(True, "resuming without --deep-think is refused")


if __name__ == "__main__":
    test_off_is_untouched()
    test_search()
    print()
    if T.FAILURES:
        print(f"{len(T.FAILURES)} FAILED: " + "; ".join(T.FAILURES))
        sys.exit(1)
    print(f"all checks passed (deep-think calls: {DEEP_CALLS}); temp dir {TMP}")

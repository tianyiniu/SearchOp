"""Offline test of the longer summary (--summary-words) under the v2 executor, and of the removal
of the deep-think speaker (2026-10-06; it was a v2-only option no pipeline used): fake models, a
tiny cluster search with 300-word summaries, and checks that nothing of deep-think is left and that
a run with the default summary is untouched.

    python tests/test_deep_think.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))   # the code under test
sys.path.insert(0, str(Path(__file__).resolve().parent))                       # the other test files

import test_pipeline_cluster as T3  # noqa: E402  (installs the fake clients on import)
import debate_mcq as D  # noqa: E402
import evolve_pipeline_cluster as V  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import program_seeds_cluster as S3  # noqa: E402
import program_space as P  # noqa: E402

T, check, run_cli, TMP = T3.T, T3.check, T3.run_cli, T3.TMP
_old_create = T.FakeCompletions.create


def _create(self, model, messages, temperature, max_tokens=None, extra_body=None, **kw):
    resp = _old_create(self, model, messages, temperature, max_tokens, extra_body)
    resp.usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=7)     # token accounting
    return resp


T.FakeCompletions.create = _create


def cache_keys(path: Path) -> list[dict]:
    return [json.loads(json.loads(l)["k"]) for l in path.open() if l.strip()]


def test_off_is_untouched():
    print("switched off")
    settings = P.configure_executor()
    before = (dict(D.PERSONA_PROMPTS), sorted(M.ACTIONS), list(P.PLAN_ROUNDS), list(P.PROTOCOLS), P.grammar_text())
    settings = P.configure_executor()
    check(set(settings) == {"v2", "digest", "answer_tokens", "eliminator", "prompts"},
          "the settings name nothing new (the prompts' signature is in every run's settings since 2026-10-06)")
    after = (dict(D.PERSONA_PROMPTS), sorted(M.ACTIONS), list(P.PLAN_ROUNDS), list(P.PROTOCOLS), P.grammar_text())
    check(before == after and not any("deep" in str(x).lower() for x in after[:4])
          and len(after[3]) == 8 and "deep" not in after[4].lower(),
          "no deep-think persona, move, plan round, protocol or grammar line")
    left = [n for mod in (D, M, P) for n in dir(mod) if "deep" in n.lower() and n != "deepcopy"]   # copy.deepcopy
    check(not left, "the deep-think code is gone (removed 2026-10-06)", str(left))
    check(D.THINKING_TAIL == 16000 and D.DEFAULT_HIGH_COST == 5,
          "the two values kept from it are unchanged (thinking tail 16,000 characters, high effort 5 turns)")
    try:
        run_cli(V, ["--deep-think", "--out", str(TMP / "never")])
        check(False, "--deep-think is no longer an option")
    except SystemExit as exc:
        check(exc.code == 2, "--deep-think is no longer an option", f"exit code {exc.code}")
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
    print("a search with 300-word summaries")
    T3.test_subsets()
    out = TMP / "deep"
    flags = ["--executor", "v2", "--summary-words", "300", "--digest-head", "1200", "--digest-tail", "1200"]
    common = ["--clusters", str(T3.CLUSTERS_V3), "--per-group", "3", "--live-cache", str(out / "rounds.jsonl"),
              "--workers", "8"] + flags
    run_cli(S3, common + ["--out", str(out / "seeds.json"), "--sanity-per-group", "1", "--min-turns", "1"])
    seeds = json.loads((out / "seeds.json").read_text())
    names = [s["name"] for s in seeds["seeds"]]
    lit = [s["name"] for s in seeds["seeds"] if s["source"] == "protocol"]
    check(len(names) == 14 and len(lit) == 8 and lit[0] == "direct",
          "14 seeds (6 model-written + 8 literature), direct first among the literature ones", str(lit))
    check(sum(s["source"] == "llm" for s in seeds["seeds"]) == 6
          and not any(s["source"] == "random" for s in seeds["seeds"]), "6 model-written seeds, no random ones")
    check(seeds["settings"].get("summary_words") == 300 and "deep_think" not in seeds["settings"],
          "the seed file records the summary length")
    run = out / "run"
    run_cli(V, common + ["--seeds", str(out / "seeds.json"), "--out", str(run), "--generations", "1"])
    lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    header, recs = lines[0], {d["key"]: d for d in lines[1:]}
    check(header["settings"].get("summary_words") == 300 and header["settings"]["digest"] == [1200, 1200],
          "the archive header records the summary length and the digest window")
    dd = next(d for d in recs.values() if d["name"] == "direct")
    check(all(v[1] == 1 for v in dd["reps"]["0"].values()), "one ordinary speaker counts as 1 turn")
    recs_c = [json.loads(l) for l in (out / "rounds.jsonl").open() if l.strip()]
    ok = all(len(r["usage"]) == len(r["responses"])
             and [u["persona"] for u in r["usage"]] == [pr[0] for pr in r["responses"]]
             and all(u["calls"] in (1, 2) and u["completion"] == 7 * u["calls"] and u["prompt"] == 100 * u["calls"]
                     for u in r["usage"]) for r in recs_c if not r["error"])
    check(ok, "every recording carries the tokens of each of its speakers (reply + summary calls)")
    keys = cache_keys(out / "rounds.jsonl")
    check(all(k.get("s") == 300 and k.get("d") == [1200, 1200] for k in keys),
          "every recording is keyed by the summary length and the digest window")
    kids = [d for d in recs.values() if d["gen"] > 0]
    check(len(kids) > 0 and all(max(v[1] for v in d["reps"]["0"].values()) <= 16 for d in recs.values()),
          "the 16-turn cap holds")
    run_cli(V, common + ["--out", str(run), "--pick-champions", "--heldout-cap", "3",
                         "--baselines", "direct,self_refine"])
    champs = json.loads((run / "champions.json").read_text())
    check(champs["baselines"] == ["direct", "self_refine"], "the baselines are scored on the held-out questions",
          str(champs["baselines"]))
    other = [a if a != "300" else "500" for a in common]
    try:
        run_cli(V, other + ["--out", str(run), "--resume", "--generations", "1"])
        check(False, "resuming with another summary length is refused")
    except SystemExit:
        check(True, "resuming with another summary length is refused")


if __name__ == "__main__":
    test_off_is_untouched()
    test_search()
    print()
    if T.FAILURES:
        print(f"{len(T.FAILURES)} FAILED: " + "; ".join(T.FAILURES))
        sys.exit(1)
    print(f"all checks passed; temp dir {TMP}")

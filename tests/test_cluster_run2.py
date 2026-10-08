"""Offline checks for the prompts of 2026-10-06 and the options of the qwen9b run2 cluster pipeline
(run_pipeline_cluster.sh qwen9b):

  the prompts              a general solver system prompt with SuperGPQA's own question prompt (the
                           external direct baseline's request), a critic that reads the whole
                           discussion, no "commit" in any prompt, the discussion round by round, the
                           prompts' signature in every cache key and in the settings (debate_mcq)
  --count-read-summaries   a debate's tokens leave out the summaries no later round read
                           (debate_mcq.SUMMARY_SPLIT, evolve_program_mcq.COUNT_READ_SUMMARIES)
  --tie-questions          slots A and G and the champion step: within t questions of the best, the
                           fewest turns wins (evolve_pipeline_cluster.strongest)
  the turn-cap record      program_space.cap_cut / run_program's "capped"
  self_refine_high         up to 2 feedback -> refine rounds, stopping after a critic that keeps
                           the answer (the kept_answer condition)
  ours against external    scripts/compare_external_baselines.py (our Direct CoT and Self-Refine
                           against the external ones)
  copying run1's seeds     scripts/copy_seeds_cluster.py (literature seeds take their current definition)

Uses the fake debate model and guide of test_pipeline_cluster.py (no vLLM, no API key). The
options are off by default: the checks also show that off, nothing changes.

    python tests/test_cluster_run2.py
"""

from __future__ import annotations

import json
import random
import sys
import types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))   # the code under test
sys.path.insert(0, str(HERE))                       # the other test files
sys.path.insert(0, str(HERE.parent / "baselines"))

import test_pipeline_cluster as T3  # noqa: E402  (installs the fake clients on import)
import test_program_clusters as T  # noqa: E402
import debate_mcq as D  # noqa: E402
import evolve_program_mcq as M  # noqa: E402
import schema_fitness as SF  # noqa: E402
import program_space as P  # noqa: E402
import evolve_pipeline_cluster as V  # noqa: E402
import compare_external_baselines as C  # noqa: E402
import copy_seeds_cluster as CS  # noqa: E402
import tasks as BT  # noqa: E402  (baselines/tasks.py: the external baselines' prompts)

check, run_cli, TMP = T.check, T.run_cli, T.TMP
QWEN = "Qwen/Qwen3.5-9B"
RUN2 = dict(executor="v3", window=65536, plain_instruction=True, last_round_vote=True,
            high_cost=3, turn_cap=15, count_read_summaries=True)
RUN2_ARGS = ["--plain-instruction", "--last-round-vote", "--high-cost", "3", "--turn-cap", "15",
             "--count-read-summaries"]


def exit_code(module, argv: list[str]) -> int:
    """run_cli, with the exit code of a main() that ends in sys.exit (0 if it returns)."""
    try:
        run_cli(module, argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


# --- the prompts ------------------------------------------------------------------------------------

def speaker_texts() -> list[str]:
    """Every fixed text a speaker can be sent under the current switches."""
    rounds = [[("solver", "Step.\nANSWER: A")], [("critic", "Flaw.\nANSWER: B")]]
    used = ("solver", "expert", "critic", "verifier", "synthesizer")     # the v3 speakers (the judge's texts below)
    return ([D.PERSONA_PROMPTS[p] for p in used] + [D.EXPERT_TMPL, SF.REWRITTEN_CRITIC, D.ANSWER_INSTR,
            D.summary_nudge_v3("A"), D.summary_nudge_v3(None), D.COMMIT_NUDGE, D.SUMMARY_NUDGE, D.COMMIT_MARK,
            D.JUDGE_PROMPT, D.judge_block([]), D.judge_block(["A", "B"]), D.judge_pick_nudge(["A"]),
            D._visible(rounds, "all", 4, [{"personas": ["solver"]}, {"personas": ["critic"]}]),
            D._visible(rounds, "letters_only", 4), D._visible(rounds, "no_letters", 4), D._where(3, 3),
            D.official_question("Q?", ["x", "y"])])


def test_prompts():
    print("the prompts")
    real = json.loads((P.ROOT / "datasets/supergpqa_600_train.json").read_text())[:60]
    check(D.SUPERGPQA_PROMPT == BT.PROMPT_TEMPLATE, "the solver's question prompt is the one the external baselines send")
    check(all(D.official_question(r["question"], r["options"]) == BT.messages(r)[0]["content"] for r in real),
          "for every question, the text equals the external direct baseline's first message")
    bad = []
    for kw in (dict(executor="v3", window=65536), dict(executor="v3", window=65536, plain_instruction=True),
               dict(executor="v3", window=65536, visible_reasoning=True), dict(executor="v2"),
               dict(executor="v3", window=65536, answers="open", judge_cache=TMP / "judge_unused.jsonl")):
        P.configure_executor(**kw)
        bad += [t[:60] for t in speaker_texts() if any(w in t.lower() for w in ("commit", "rival", "survive",
                                                                                  "other participants"))]
        check(D.PERSONA_PROMPTS["solver"] == D.SOLVER_SYSTEM and D.SOLVER_SYSTEM.startswith("You are a helpful assistant")
              and "step by step" not in D.SOLVER_SYSTEM,
              f"the solver's system prompt is the general one ({kw.get('executor')}, {kw.get('answers', 'letters')})")
        check(SF.REWRITTEN_CRITIC.startswith(D.CRITIC_BODY_OPEN if D.OPEN else D.CRITIC_BODY)
              and D.PERSONA_PROMPTS["critic"].startswith(D.CRITIC_BODY_OPEN if D.OPEN else D.CRITIC_BODY)
              and "whole discussion" in SF.REWRITTEN_CRITIC, "both critics read the whole discussion")
    check(not bad, "no text a speaker is sent says 'commit', 'rival', 'survive' or 'other participants'", str(bad))
    P.configure_executor(executor="v3", window=65536, visible_reasoning=True)
    P.configure_executor(executor="v3", window=65536)
    check(not D.VISIBLE_REASONING, "a later configure turns visible reasoning off again")

    on = P.configure_executor(**RUN2)
    sig = D.prompt_signature()
    check(on.get("prompts") == sig and on.get("count_read_summaries") is True and on["high_cost"] == 3
          and on["turn_cap"] == 15 and P.MAX_TURNS == 15, "run2's settings name the prompts' signature and the budget")
    check(json.loads(SF.blind_key("solver", "high", 0)).get("t") == sig
          and json.loads(SF.path_key([P.PLAN_ROUNDS["critic|high"]])).get("t") == sig,
          "every cache key carries the prompts' signature")
    old = D.PERSONA_PROMPTS
    D.PERSONA_PROMPTS = {**old, "verifier": old["verifier"] + " Be brief."}
    changed = D.prompt_signature()
    D.PERSONA_PROMPTS = old
    check(changed != sig and D.prompt_signature() == sig, "a change to any persona prompt changes the signature")
    off = P.configure_executor(executor="v3", window=65536)
    check(off["prompts"] != sig and "count_read_summaries" not in off,
          "the plain instruction gives another signature; the summary option is named only when on")
    P.configure_executor(**RUN2)

    row = real[0]
    n = len(row["options"])
    c = T3._Scripted(T3._Resp("The options compare as follows.\nAnswer: C"))
    out = D.execute_round(c, QWEN, row["question"], row["options"], {"personas": ["solver"], "effort": "high"},
                          [], 1.0)
    req = c.requests[0]
    check(req["messages"] == [{"role": "system", "content": D.SOLVER_SYSTEM + " Think step by step. Explain your "
                               "reasoning in your response before you give the final answer."}] + BT.messages(row),
          "a blind high-effort solver: the general system prompt with 'Think step by step.' and the request to "
          "explain its reasoning, then exactly the external direct request",
          json.dumps(req["messages"])[:200])
    check(req["temperature"] == 1.0 and req["top_p"] == 0.95 and req["presence_penalty"] == 1.5
          and req["extra_body"] == {"top_k": 20, "min_p": 0.0, "chat_template_kwargs": {"enable_thinking": True}},
          "... with the baseline's sampling (temperature 1, top_p 0.95, top_k 20, presence 1.5, thinking on)",
          str(req))
    check(D.extract_letter(out[0][1], n) == "C" and len(c.requests) == 1,
          "its 'Answer: X' line is read as the letter; a short reply with a letter needs no other call")
    long = "One long derivation step. " * 150 + "\nAnswer: A"
    c = T3._Scripted(T3._Resp(long), T3._Resp("Step two decides it.\nANSWER: A"))
    D.execute_round(c, QWEN, row["question"], row["options"], {"personas": ["solver"]}, [], 1.0)
    check(len(c.requests) == 2 and c.requests[0]["messages"][0] == {
              "role": "system", "content": D.SOLVER_SYSTEM + " Explain your reasoning in your response before you "
                                                             "give the final answer."}
          and c.requests[1]["messages"][0] == c.requests[0]["messages"][0],
          "a low-effort solver: the general system prompt, asked to explain its reasoning in the reply (the "
          "baselines' sentence); its summary call the same")
    check(D.SOLVER_EXPLAIN.strip() == BT.EXPLAIN_SENTENCE, "the sentence is the external baselines' own")


def test_every_request():
    """Every speaker, at both efforts, first in the debate, after a discussion, and blind in a later
    round: the request asks for the reasoning in the reply, has the answer format, says neither
    'commit' nor 'other participants', shows the discussion exactly when the speaker sees it, and a
    reviewer that speaks first is told there is nothing to review."""
    print("every request, every case")
    P.configure_executor(**RUN2)
    row = json.loads((P.ROOT / "datasets/supergpqa_600_train.json").read_text())[0]
    prior = [[("solver", "Reasoning of solver 1.\nANSWER: B")]]
    reason = ("explain", "show each check")     # to write it out in the reply, not only to think it
    bad, n = [], 0
    for p in ["solver", "expert", "critic", "verifier", "synthesizer"]:
        for eff in ("low", "high"):
            cases = [("first", [], {"personas": [p], "effort": eff}),
                     ("sees", prior, {"personas": [p], "effort": eff})]
            if p in ("solver", "expert"):
                cases.append(("blind", prior, {"personas": [p], "effort": eff, "sees": "none"}))
            for where, rounds, spec in cases:
                c = T3._Scripted(T3._Resp("Short reasoning.\nANSWER: A"))
                D.execute_round(c, QWEN, row["question"], row["options"], spec, rounds, 1.0,
                                prompts=M.B.question_prompts(row),
                                prior_specs=[{"personas": ["solver"]}] if rounds else None)
                text = " ".join(m["content"] for m in c.requests[0]["messages"]).lower()
                n += 1
                ok = (any(k in text for k in reason) and "answer:" in text and "commit" not in text
                      and "other participants" not in text and ("the discussion so far" in text) == (where == "sees")
                      and ("no one has answered this question yet" in text) == (p in D.REVIEWERS and where == "first"))
                if not ok:
                    bad.append(f"{p} {eff} {where}")
    check(not bad and n == 24, "all 24 cases: reasoning asked for, answer format, no banned words, the discussion "
                               "only when seen, reviewers told when there is nothing to review", str(bad))
    check("Show each check in your response." in D.PERSONA_PROMPTS["verifier"]
          and "Explain your choice in your response." in D.PERSONA_PROMPTS["synthesizer"],
          "the verifier and the synthesizer are asked to write out their reasoning")
    check(all(w not in D.PERSONA_PROMPTS["verifier"] for w in ("rival", "survive")),
          "the verifier says neither 'rival' nor 'survives'")
    expert = M.B.question_prompts(row)["expert"]
    low, high = D.speaker_system("expert", expert, "low"), D.speaker_system("expert", expert, "high")
    check("step by step" not in low.lower() and D.EXPERT_LOW in low and D.EXPERT_STEP in high,
          "the expert reasons step by step only at high effort")
    c = T3._Scripted(T3._Resp("x\nANSWER: A"))
    D.execute_round(c, QWEN, row["question"], row["options"], {"personas": ["expert"]}, [], 1.0,
                    prompts=M.B.question_prompts(row))
    check(c.requests[0]["messages"][0]["content"] == low, "a low-effort expert call sends the low-effort text")


def test_discussion():
    print("the discussion, round by round")
    P.configure_executor(**RUN2)
    rounds = [[("solver", "Reason A.\nANSWER: B"), ("solver", "Reason B.\nAnswer: C")],
              [("critic", "A flaw in step two.\n**ANSWER: B**")],
              [("expert", "The expert view.\nANSWER: D")]]
    specs = [{"personas": ["solver", "solver"]}, {"personas": ["critic"]}, {"personas": ["expert"], "sees": "none"}]
    want = (D.DISCUSSION_HEAD + "\n\n"
            "Round 1 (2 speakers; they saw only the question)\n\n"
            "[Round 1, solver 1] Answer: B\nReason A.\n\n[Round 1, solver 2] Answer: C\nReason B.\n\n"
            "Round 2 (1 speaker; it saw round 1)\n\n[Round 2, critic] Answer: B\nA flaw in step two.\n\n"
            "Round 3 (1 speaker; it saw only the question)\n\n[Round 3, expert] Answer: D\nThe expert view.")
    got = D._visible(rounds, "all", 4, specs)
    check(got == want, "rounds, what each round saw, each speaker's answer, the reasoning without its answer line",
          got)
    check(D._visible(rounds, "all", 4).startswith(D.DISCUSSION_HEAD + "\n\nRound 1 (2 speakers)\n"),
          "without the specs, what a round saw is left out")
    four = specs + [{"personas": ["solver", "solver"]}]
    check("Round 4 (2 speakers; they saw rounds 1 to 3)" in D._visible(rounds + [[("solver", "x\nANSWER: A"),
                                                                                ("solver", "y\nANSWER: B")]],
                                                                       "all", 4, four),
          "a round that saw several rounds names them")
    long_summary = "x " * 600 + "\nANSWER: A" + D.SUMMARY_MARK + "The summary.\nANSWER: A"
    check("[Round 1, solver] Answer: A\nThe summary." in D._visible([[("solver", long_summary)]], "all", 4,
                                                                  [{"personas": ["solver"]}]),
          "a summarised reply shows its summary")

    row = {"id": "q", "question": "Which?", "options": ["a", "b", "c", "d"], "answer_letter": "A"}
    c = T3._Scripted(T3._Resp("Weighing them.\nANSWER: B"))
    D.execute_round(c, QWEN, row["question"], row["options"], {"personas": ["synthesizer"]}, rounds, 1.0,
                    prior_specs=specs)
    user = c.requests[0]["messages"][1]["content"]
    check(user.startswith(D.render_question("Which?", row["options"]) + "\n\n" + D.DISCUSSION_HEAD)
          and user.endswith("You speak in round 4. Give your response. " + D.ANSWER_INSTR),
          "a speaker that sees the discussion is told which round it speaks in", user[-160:])
    c = T3._Scripted(*[T3._Resp(f"Mine.\nANSWER: {l}") for l in "ABC"])
    D.execute_round(c, QWEN, row["question"], row["options"], {"personas": ["solver"] * 3}, rounds, 1.0,
                    prior_specs=specs)
    user = c.requests[0]["messages"][1]["content"]
    check(user.startswith(D.official_question("Which?", row["options"]) + "\n" + D.DISCUSSION_HEAD)
          and user.endswith("You speak in round 4. 2 other speakers answer in this round at the same time; you do "
                            "not see their replies. Give your response."),
          "a solver that sees the discussion: its own prompt, the discussion, the round and its other speakers",
          user[-200:])
    c = T3._Scripted(T3._Resp("Alone.\nANSWER: A"))
    D.execute_round(c, QWEN, row["question"], row["options"], {"personas": ["expert"], "sees": "none"}, rounds, 1.0,
                    prior_specs=specs)
    check("Round" not in c.requests[0]["messages"][1]["content"], "a blind speaker sees no discussion and no round")

    # the round runner passes the specs: a critic after two blind solvers, through the cache
    seen: list[str] = []
    real_chat = D.chat_v3

    def spy(client, model, system, user, n, effort="low", info=None):
        seen.append(user)
        return real_chat(client, model, system, user, n, effort, info)
    D.chat_v3 = spy
    try:
        rows = {r["id"]: r for r in json.loads(T.DATASET.read_text())}
        q = sorted(rows)[0]
        runner = P.make_runner(rows, TMP / "v3r2_disc.jsonl", model=QWEN, progress=False)
        runner.reset_budget(None)
        prog = P.normalize_program({"plan": [P.PLAN_ROUNDS["solver_x2"], P.PLAN_ROUNDS["critic"]],
                                    "rules": [P.CONT()], "default": "stop:last_commit"})
        M.run_program(prog, runner, rows[q], rep=0, max_calls=P.MAX_TURNS)
        runner.close()
    finally:
        D.chat_v3 = real_chat
    critic = [u for u in seen if D.DISCUSSION_HEAD in u]
    check(len(critic) == 1 and "Round 1 (2 speakers; they saw only the question)" in critic[0]
          and "You speak in round 2." in critic[0], "in a run, the critic sees round 1's structure", str(len(critic)))


# --- counting only the summaries a later round read ------------------------------------------------

def test_read_summaries():
    print("summaries counted only when read")
    S, A, N, L = ({"personas": ["solver"], "effort": "high"}, {"personas": ["critic"]},
                  {"personas": ["solver"], "sees": "none"}, {"personas": ["verifier"], "sees": "last_round"})
    check(M.round_read_later([S, A], 0) and not M.round_read_later([S, A], 1), "a round that sees all reads the rounds before it")
    check(not M.round_read_later([S, N], 0), "a blind round reads nothing")
    check(not M.round_read_later([S, N, L], 0) and M.round_read_later([S, N, L], 1),
          "a round that sees the last round reads only the round just before it")
    check(not M.round_read_later([S, {"personas": ["critic"], "sees": "letters_only"}], 0),
          "a round that sees only the letters reads no summary")
    check(not M.round_read_later([S, {"personas": ["independent"]}], 0), "a speaker that ignores the debate reads nothing")

    rows = {r["id"]: r for r in json.loads(T.DATASET.read_text())}
    qids = sorted(rows)[:12]
    progs = {"direct_high": P.PROTOCOLS["direct_high"], "self_refine_high": P.PROTOCOLS["self_refine_high"],
             "blind_pair": {"plan": [P.PLAN_ROUNDS["solver|high"], dict(P.PLAN_ROUNDS["solver|high"], sees="none")],
                            "rules": [P.CONT()], "default": "stop:vote"}}
    progs = {k: P.normalize_program(v) for k, v in progs.items()}

    def run_all(cache: Path, settings: dict):
        P.configure_executor(**settings)
        runner = P.make_runner(rows, cache, model=QWEN, progress=False)
        runner.reset_budget(None)
        out = {(name, q): M.run_program(prog, runner, rows[q], rep=0, max_calls=P.MAX_TURNS)
               for name, prog in progs.items() for q in qids}
        parts = {}                   # per debate: the rounds it ran (from its path), their tokens, their summaries'
        for (name, q), o in out.items():
            prompts = M.B.question_prompts(rows[q])
            specs, full, summ = [], [], []
            prog = progs[name]
            k = 0
            for act in o["actions"]:
                for spec in ([prog["plan"][k]] if act == "continue" else M.ACTIONS[act]):
                    full.append(runner.round_tokens(q, specs, spec, 0, prompts))
                    summ.append(runner.round_summary_tokens(q, specs, spec, 0, prompts))
                    specs.append(spec)
                k += act == "continue"
            parts[(name, q)] = (specs, full, summ)
        runner.close()
        return out, parts

    split_settings = dict(executor="v3", window=65536, count_read_summaries=True)
    out_on, parts = run_all(TMP / "v3r2_summ.jsonl", split_settings)
    ok, dropped = True, Counter()
    for key, o in out_on.items():
        specs, full, summ = parts[key]
        want = sum(full) - sum(s for i, s in enumerate(summ) if not M.round_read_later(specs, i))
        ok &= o["tokens"] == want
        dropped[key[0]] += sum(full) - o["tokens"]
        if key[0] == "self_refine_high":
            ok &= all(summ[i] == 0 or M.round_read_later(specs, i) for i in range(len(specs) - 1))
    check(ok, "a debate's tokens = all its tokens less the summaries no later round read")
    check(all(dropped[k] > 0 for k in progs), "every program had some unread summary left out", str(dict(dropped)))
    lines = [json.loads(l) for l in (TMP / "v3r2_summ.jsonl").open()]
    check(any("summary" in u for d in lines for u in d.get("usage", [])), "the recordings keep summary tokens apart")
    check(all(u.get("summary", 0) <= u["completion"] for d in lines for u in d.get("usage", [])),
          "a summary's tokens are part of its speaker's tokens")

    out_off, parts_off = run_all(TMP / "v3r2_summ.jsonl", dict(executor="v3", window=65536))
    check(all(out_off[k]["tokens"] == sum(parts_off[k][1]) for k in out_off),
          "with the option off the same recordings count in full")
    check(all(out_off[k]["letter"] == out_on[k]["letter"] and out_off[k]["actions"] == out_on[k]["actions"]
              for k in out_on), "and the answers and paths are the same: only the count changes")
    out_old, parts_old = run_all(TMP / "v3r2_nosplit.jsonl", dict(executor="v3", window=65536))
    out_new, _ = run_all(TMP / "v3r2_nosplit.jsonl", split_settings)
    check(all(out_new[k]["tokens"] == sum(parts_old[k][1]) for k in out_new),
          "a recording made without the split counts in full even with the option on (never too few)")
    P.configure_executor(executor="v3", window=65536)


# --- the turn-cap record --------------------------------------------------------------------------

def test_cap_record():
    print("debates the turn cap stopped")
    P.configure_executor(executor="v3", window=65536, high_cost=3, turn_cap=10)
    row = {"id": "q", "question": "?", "options": ["a", "b", "c", "d"], "answer_letter": "A"}

    class Runner:
        def __init__(self, seed):
            self.rng = random.Random(seed)

        def run_round(self, qid, rounds, specs, spec, rep=0, prompts=None):
            return [(p, f"ANSWER: {self.rng.choice('ABCD')}") for p in spec["personas"]]

    # three high-effort rounds with no early stop (self_refine_high may stop by its own rule)
    sr = P.normalize_program({"plan": [P.PLAN_ROUNDS["solver|high"], P.PLAN_ROUNDS["critic|high"],
                                       P.PLAN_ROUNDS["solver|high"]], "rules": [P.CONT()], "default": "stop:last_commit"})
    rec = P.ProgRecord(sr, "sr", "sr", 0)
    out = M.run_program(sr, Runner(0), row, max_calls=6)
    rec.record(0, "q", out)
    check(out["capped"] and P.cap_cut(sr, rec.reps[0]["q"]) and out["n_calls"] == 6,
          "a plan over the cap: stopped after 6 of its 9 turns, and both records say so")
    out = M.run_program(sr, Runner(0), row, max_calls=10)
    rec.record(0, "q", out)
    check(not out["capped"] and not P.cap_cut(sr, rec.reps[0]["q"]), "within the cap: not stopped by it")
    rng, agree, cut = random.Random(5), 0, 0
    progs = [P.random_program(rng) for _ in range(300)]
    for i, prog in enumerate(progs):
        r = P.ProgRecord(prog, f"p{i}", "x", 0)
        for cap in (3, 6, 10):
            o = M.run_program(prog, Runner(i * 7 + cap), row, max_calls=cap)
            r.record(0, "q", o)
            agree += o["capped"] == P.cap_cut(prog, r.reps[0]["q"])
            cut += o["capped"]
    check(agree == 900 and cut > 50, "on 300 random programs at three caps, the record and the run agree",
          f"{agree}/900 agree, {cut} cut")
    check(not P.cap_cut(sr, [1, 3, "B", "continue", "", 100]), "a record without rules fired is never called cut")
    P.configure_executor(executor="v3", window=65536)


# --- the one-question tie rule ---------------------------------------------------------------------

def test_tie_rule():
    print("ties within one question")
    progs = [types.SimpleNamespace(name=n, gen=g, s=s, t=t) for n, g, s, t in
             (("a", 0, 0.60, 9.0), ("b", 1, 0.59, 3.0), ("c", 2, 0.58, 1.0), ("d", 3, 0.60, 9.0))]
    pick = lambda tie: V.strongest(progs, lambda r: r.s, lambda r: r.t, 100, tie).name
    check(pick(0) == "a", "tie 0: the best score, then the older program (the earlier rule)")
    check(pick(1) == "b" and pick(2) == "c", "tie 1: within one question, the fewest turns; tie 2: within two")
    old = min(progs, key=lambda r: (-round(r.s, 9), r.t, r.gen, r.name))
    check(V.strongest(progs, lambda r: r.s, lambda r: r.t, 100, 0) is old, "tie 0 is the earlier expression exactly")
    check(V.tie_questions(types.SimpleNamespace()) == 0 and V.tie_questions(types.SimpleNamespace(tie_questions=1)) == 1,
          "a namespace without the option means exact ties")


def test_search_run2(seeds_path: Path):
    """A small search under all of run2's options, then its champion step."""
    print("a search with run2's options")
    run, cache = TMP / "v3r2" / "run", TMP / "v3r2" / "rounds.jsonl"
    # the fake seeds were written for the default budget; keep those whose plans fit run2's (as run1's do)
    P.configure_executor(**RUN2)
    seeds = json.loads(seeds_path.read_text())
    seeds["seeds"] = [s for s in seeds["seeds"] if P.plan_cost(P.normalize_program(s["program"])) <= P.MAX_TURNS]
    seeds_path = TMP / "v3r2" / "seeds.json"
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    seeds_path.write_text(json.dumps(seeds))
    check(sum(s["source"] == "protocol" for s in seeds["seeds"]) == 8, "all 8 literature seeds fit the 15-turn cap",
          str(len(seeds["seeds"])))
    P.configure_executor(executor="v3", window=65536)
    common = ["--seeds", str(seeds_path), "--out", str(run), "--clusters", str(T3.CLUSTERS_V3),
              "--per-group", "3", "--live-cache", str(cache), "--workers", "8"] + T3.WINDOW + RUN2_ARGS
    run_cli(V, common + ["--tie-questions", "1", "--generations", "2"])
    lines = [json.loads(l) for l in (run / "archive.jsonl").open() if l.strip()]
    header, recs = lines[0], {}
    for d in lines[1:]:
        recs[d["key"]] = d
    st = header["settings"]
    check(header.get("tie_questions") == 1 and st.get("prompts") and st.get("count_read_summaries")
          and st["high_cost"] == 3 and st["turn_cap"] == 15, "the header names the tie rule and run2's settings",
          json.dumps(st))
    check(all(P.plan_cost(d["program"]) <= 15 for d in recs.values()), "no plan over the 15-turn cap")
    cached = [json.loads(l) for l in cache.open()]
    check(all(json.loads(d["k"]).get("t") == st["prompts"] for d in cached),
          "every recording carries the prompts' signature of the run's settings")
    check(any(json.loads(d["k"]).get("blind", [""])[0] == "solver" for d in cached),
          "blind solvers were recorded under those keys")
    gens = [json.loads(l) for l in (run / "generations.jsonl").open() if l.strip()]
    check(all("cut_by_cap" in s for g in gens for s in g["slots"]), "the slot tables count debates the cap stopped")

    s = V.Search(types.SimpleNamespace(out=str(TMP / "v3r2" / "check"), seed=0, max_calls_per_question=15,
                                       tie_questions=1),
                 {}, P.load_groups(T3.CLUSTERS_V3, 3), None, {})
    for d in recs.values():
        r = P.ProgRecord.from_json(d)
        r.dup_of = None
        s.archive[r.key] = r
    for r in s.archive.values():
        s.mark_duplicate(r)
    slots = s.compute_slots()
    logged = {(x["group"], x["slot"]): x["key"] for x in gens[-1]["slots"]}
    check(slots == logged, "the logged slots follow from the archive under the tie rule")
    pool = s.eligible()
    ok = True
    for g in s.group_ids + [V.GLOBAL]:
        k = slots[(g, "A" if g != V.GLOBAL else "G")]
        n = len(s.gq[g])
        best = max(s.score(r, g) for r in pool)
        h = s.archive[k]
        ok &= s.score(h, g) >= best - 1 / n - 1e-9
        ok &= all(s.turns(r, g) >= s.turns(h, g) for r in pool if s.score(r, g) >= best - 1 / n - 1e-9)
    check(ok, "each A and G holder is within one question of the best, with the fewest turns of those")
    exact = V.Search(types.SimpleNamespace(out=str(TMP / "v3r2" / "check0"), seed=0, max_calls_per_question=15),
                     {}, P.load_groups(T3.CLUSTERS_V3, 3), None, {})
    exact.archive, exact.by_behaviour = s.archive, s.by_behaviour
    e_slots = exact.compute_slots()
    check(all(abs(exact.score(exact.archive[e_slots[(g, "A")]], g) - max(exact.score(r, g) for r in pool)) < 1e-9
              for g in exact.group_ids), "without the option, slot A is the best score, as before")
    try:
        run_cli(V, common + ["--generations", "2", "--resume"])
        check(False, "a resume with another tie rule is refused")
    except SystemExit:
        check(True, "a resume with another tie rule is refused")
    run_cli(V, common + ["--tie-questions", "1", "--pick-champions", "--heldout-cap", "4",
                         "--baselines", "direct_high,self_refine_high"])
    champs = json.loads((run / "champions.json").read_text())
    ok = True
    for res in list(champs["per_group"].values()) + [champs["global"]]:
        fin = [p for p in res["programs"] if p["finalist"]]
        nq = res["held_out_n"]
        best = max(p["held_out_acc"] for p in fin)
        champ = next(p for p in fin if p["key"] == res["champion"])
        near = [p for p in fin if p["held_out_acc"] >= best - 1 / nq - 1e-9]
        ok &= champ in near and (champ["turns"] or 0) == min((p["turns"] or 0) for p in near)
    check(ok, "each champion: within one held-out question of the best finalist, the fewest turns")
    # run2's champion step runs the finalists only (2026-10-07): the same champions, no baseline rows
    run_cli(V, common + ["--tie-questions", "1", "--pick-champions", "--heldout-cap", "4", "--baselines", ""])
    alone = json.loads((run / "champions.json").read_text())
    pairs = list(zip([*champs["per_group"].values(), champs["global"]], [*alone["per_group"].values(), alone["global"]]))
    check(alone["baselines"] == [] and all(not p["baseline"] or p["finalist"] for _, r in pairs for p in r["programs"])
          and all(a["champion"] == b["champion"] for a, b in pairs)
          and all({p["key"] for p in b["programs"]} == {p["key"] for p in a["programs"] if p["finalist"]} for a, b in pairs),
          "without baselines the champion step runs the same finalists and picks the same champions")


# --- the solver check ------------------------------------------------------------------------------

def test_self_refine_stop():
    print("self_refine_high: up to 2 feedback -> refine rounds, stopping after a critic that keeps the answer")
    P.configure_executor(**RUN2)
    sr = P.normalize_program(P.PROTOCOLS["self_refine_high"])
    P.validate_program(sr)
    check([M.round_kind(x) for x in sr["plan"]] == ["solver", "critic", "solver", "critic", "solver"]
          and P.plan_cost(sr) == 15,
          "at 3 turns a speaker and a 15-turn cap: 2 feedback -> refine rounds (15 turns)")
    row = {"id": "q", "question": "?", "options": ["a", "b", "c", "d"], "answer_letter": "A"}

    class Runner:
        def __init__(self, letters):
            self.letters = iter(letters)

        def run_round(self, qid, rounds, specs, spec, rep=0, prompts=None):
            return [(p, f"ANSWER: {next(self.letters)}") for p in spec["personas"]]
    for letters, turns, final, why in (("AA", 6, "A", "the first critic keeps the answer: stop (2 speakers)"),
                                       ("ABCC", 12, "C", "the second critic keeps the refined answer: stop (4)"),
                                       ("ABCDA", 15, "A", "neither critic keeps it: both refine rounds run (5)"),
                                       ("ABBCD", 15, "D", "a refined answer equal to the critic's is no stop: "
                                                          "only a critic's round can stop it")):
        out = M.run_program(sr, Runner(letters), row, max_calls=P.MAX_TURNS)
        check(out["n_calls"] == turns and out["letter"] == final and not out["capped"], why, str(out))
    P.configure_executor(executor="v3", window=65536)
    one = P.normalize_program(P.PROTOCOLS["self_refine_high"])
    check(len(one["plan"]) == 3 and P.plan_cost(one) == 15,
          "at 5 turns a speaker and the 16-turn cap: 1 feedback -> refine round (15 turns)")


def test_against_external():
    print("ours against the external baselines")
    rows = json.loads(T.DATASET.read_text())[:20]
    qfile, cache = TMP / "v3r2_check_q.json", TMP / "v3r2_check_rounds.jsonl"
    qfile.write_text(json.dumps(rows))
    k3_d, raw_d, k3_s = TMP / "v3r2_ext_direct_k3.json", TMP / "v3r2_ext_direct_raw.jsonl", TMP / "v3r2_ext_sr_k3.json"

    def externals(t_direct: int, t_sr: int, marks: list[int], cut: int = 0):
        for path, t in ((k3_d, t_direct), (k3_s, t_sr)):
            path.write_text(json.dumps({"report": {}, "per_question": {
                r["id"]: {"n": 3, "marks": marks, "preds": ["A"] * 3, "answer": "A", "tokens": [t] * 3}
                for r in rows}}))
        with raw_d.open("w") as fh:
            for i, r in enumerate(rows):
                for s in range(3):
                    fh.write(json.dumps({"id": r["id"], "sample_idx": s, "error": None,
                                         "finish_reason": "length" if i * 3 + s < cut else "stop",
                                         "completion_tokens": t_direct}) + "\n")

    out = TMP / "v3r2_check" / "external_baselines"
    argv = ["--questions", str(qfile), "--out", str(out), "--external-direct", str(k3_d),
            "--external-direct-raw", str(raw_d), "--external-selfrefine", str(k3_s), "--model", QWEN,
            "--live-cache", str(cache), "--workers", "4", "--context-window", "65536"] + RUN2_ARGS
    externals(1, 1, [1, 1, 1])
    n0 = T.MODEL_CALLS["chat"]
    code = exit_code(C, argv)
    res = json.loads(out.with_suffix(".json").read_text())
    ok = {c["name"]: c["ok"] for c in res["checks"]}
    check(code == 1 and res["verdict"] == "fail" and not ok["Direct CoT: accuracy"]
          and not any("tokens" in name for name in ok)
          and ok["Direct CoT: every question ran"] and ok["Self-Refine: every question ran"]
          and ok["low-effort solver ran"], "external baselines always right: the accuracy check fails (exit 1); "
          "token use is not a check", json.dumps(res["checks"])[:400])
    md = out.with_suffix(".md").read_text()
    check("## Token use (not a check)" in md and "| Self-Refine (self_refine_high) |" in md.split("## Token use")[1],
          "the report shows each program's tokens against the external program's")
    check(T.MODEL_CALLS["chat"] > n0 and res["settings"].get("prompts") and out.with_suffix(".md").exists(),
          "the report is written", str(T.MODEL_CALLS["chat"] - n0))
    prog = {p["ours"]: p for p in res["programs"]}
    check(6.0 <= prog["self_refine_high"]["turns"] <= 15.0, "Self-Refine ran 6 to 15 turns per debate",
          str(prog["self_refine_high"]["turns"]))
    externals(round(prog["direct_high"]["tokens"]), round(prog["self_refine_high"]["tokens"]), [0, 0, 0], cut=3)
    n1 = T.MODEL_CALLS["chat"]
    code = exit_code(C, argv)
    res = json.loads(out.with_suffix(".json").read_text())
    check(code == 0 and res["verdict"] == "pass" and T.MODEL_CALLS["chat"] == n1,
          "matching tokens and accuracy: the check passes, replaying its own debates for free",
          json.dumps(res["checks"])[:400])
    check(all(len(v["right"]) == 3 for p in ("direct_high", "self_refine_high") for v in res["per_question"][p].values())
          and all(p["external_runs"] == 3 for p in res["programs"]),
          "three external runs: ours run three times too (replicates 0, 1, 2)")
    check(abs(res["external_direct_cut_off_share"] - 3 / 60) < 1e-9, "the baseline's cut-off share is read")
    externals(1, 1, [0, 0, 0], cut=3)                     # tokens far from ours: no longer a reason to fail
    code = exit_code(C, argv)
    res = json.loads(out.with_suffix(".json").read_text())
    check(code == 0 and res["verdict"] == "pass" and T.MODEL_CALLS["chat"] == n1,
          "tokens far from the external baselines' do not fail the comparison")
    # generation 0 of the search runs every seed at replicates 0 and 1: it replays the check's debates
    P.configure_executor(**RUN2)
    by_id = {r["id"]: r for r in rows}
    runner = P.make_runner(by_id, cache, model=QWEN, progress=False)
    n2 = T.MODEL_CALLS["chat"]
    outs = [M.run_program(P.normalize_program(P.PROTOCOLS[name]), runner, by_id[r["id"]], rep=rep,
                          max_calls=P.MAX_TURNS)
            for name in ("direct_high", "self_refine_high") for rep in (0, 1) for r in rows]
    runner.close()
    check(T.MODEL_CALLS["chat"] == n2 and all(o["letter"] for o in outs),
          "the search's direct_high and self_refine_high seeds (replicates 0 and 1) replay with no model call")
    # one external run (HLE's comparison): ours run once too, at replicate 0, on a fresh round cache;
    # generation 0 then replays replicate 0 and runs replicate 1 (not in the cache: a replay-only runner
    # meets it off the cache)
    cache1 = TMP / "v3r2_check_rounds_one.jsonl"
    cache1.unlink(missing_ok=True)
    for path in (k3_d, k3_s):
        path.write_text(json.dumps({"report": {}, "per_question": {
            r["id"]: {"n": 1, "marks": [1], "preds": ["A"], "answer": "A", "tokens": [1]} for r in rows}}))
    raw_d.write_text("".join(json.dumps({"id": r["id"], "sample_idx": 0, "error": None, "finish_reason": "stop",
                                         "completion_tokens": 1}) + "\n" for r in rows))
    n3 = T.MODEL_CALLS["chat"]
    exit_code(C, [str(cache1) if a == str(cache) else a for a in argv])
    res = json.loads(out.with_suffix(".json").read_text())
    ran = {p: sorted({len(v["right"]) for v in res["per_question"][p].values()}) for p in ("direct_high", "self_refine_high")}
    check(ran == {"direct_high": [1], "self_refine_high": [1]} and T.MODEL_CALLS["chat"] > n3
          and all(p["external_runs"] == 1 for p in res["programs"])
          and "1 run per question each" in out.with_suffix(".md").read_text(),
          "one external run: ours run once too (replicate 0), and the report says so", str(ran))
    P.configure_executor(**RUN2)
    runner = P.make_runner(by_id, cache1, model=QWEN, progress=False)
    before = T.MODEL_CALLS["chat"]
    outs = [M.run_program(P.normalize_program(P.PROTOCOLS[name]), runner, by_id[r["id"]], rep=0,
                          max_calls=P.MAX_TURNS)
            for name in ("direct_high", "self_refine_high") for r in rows]
    check(T.MODEL_CALLS["chat"] == before and all(o["letter"] for o in outs),
          "generation 0, replicate 0: replayed from the check with no model call")
    off = 0
    for name in ("direct_high", "self_refine_high"):
        for r in rows:
            try:
                M.run_program(P.normalize_program(P.PROTOCOLS[name]), runner, by_id[r["id"]], rep=1,
                              max_calls=P.MAX_TURNS)
            except M.OffCache:
                off += 1
    check(off == 2 * len(rows), "generation 0, replicate 1: not run by the check (the search runs it)",
          f"{off} of {2 * len(rows)} off the cache")
    runner.close()
    P.configure_executor(executor="v3", window=65536)


def test_copy_seeds(seeds_path: Path):
    print("copying an earlier run's seeds")
    src = json.loads(seeds_path.read_text())
    old_sr = {"plan": [{"personas": ["solver"], "effort": "high"}, {"personas": ["critic"], "effort": "high"},
                       {"personas": ["solver"], "effort": "high"}], "rules": [P.CONT()], "default": "stop:last_commit"}
    for s_ in src["seeds"]:
        if s_["name"] == "self_refine_high":
            s_["program"] = old_sr                    # as run1's seeds hold it
    src["seeds"] = [s_ for s_ in src["seeds"] if s_["source"] == "protocol"]   # the fakes may not fit the cap
    source, out = TMP / "v3r2_copy" / "src.json", TMP / "v3r2_copy" / "seeds.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(src))
    run_cli(CS, ["--source", str(source), "--out", str(out), "--context-window", "65536"] + RUN2_ARGS)
    got = json.loads(out.read_text())
    sr = next(s_ for s_ in got["seeds"] if s_["name"] == "self_refine_high")
    check(P.canon(P.normalize_program(sr["program"])) == P.canon(P.normalize_program(P.PROTOCOLS["self_refine_high"]))
          and got["copied_from"]["literature_seeds_redefined"] == ["self_refine_high"]
          and got["settings"]["turn_cap"] == 15, "literature seeds take their current definition; the change is named")
    out.write_text("{}")
    run_cli(CS, ["--source", str(source), "--out", str(out), "--context-window", "65536"] + RUN2_ARGS)
    check(out.read_text() == "{}", "an existing copy is kept as it is")
    P.configure_executor(executor="v3", window=65536)


def test_audit_fixes():
    """Each defect the 2026-10-06 audit found, checked (scripts/... named in each line)."""
    print("audit fixes of 2026-10-06")
    P.configure_executor(executor="v3", window=32768, plain_instruction=True, last_round_vote=True,
                         count_read_summaries=True, high_cost=3, turn_cap=15)
    norm = P.normalize_program
    row = {"id": "pattern", "question": "?", "options": [f"option {i}" for i in range(10)], "answer_letter": "A",
           "discipline": "Science", "field": "Physics", "subfield": "Physics"}

    class Pattern:                                     # every speaker answers by (round, speaker) -> letter
        def __init__(self, f):
            self.f = f

        def run_round(self, qid, rounds, specs, spec, rep=0, prompts=None):
            return [(p, f"Reasoning.\nANSWER: {self.f(len(rounds), j)}") for j, p in enumerate(spec["personas"])]
    agree = Pattern(lambda r, j: "A")

    # 1. the step limit (evolve_program_mcq.run_program): the turn cap alone ends a debate, and a cut is recorded
    loop = norm({"plan": [P.PLAN_ROUNDS["solver"]], "rules": [P.CONT(), {"when": ["step>=1"], "do": "solver|blind"}],
                 "default": "stop:vote"})
    out = M.run_program(loop, agree, row, max_calls=P.MAX_TURNS, grade=False)
    check(out["n_calls"] == 15 and out["capped"] and out["correct"] is None,
          "a loop of 1-turn rounds runs to the 15-turn cap (not 12 rounds) and is recorded as cut",
          f"{out['n_calls']} turns, capped {out['capped']}")
    out = M.run_program(loop, agree, row, max_calls=None, grade=False)
    check(out["n_calls"] == M.MAX_STEPS and out["capped"], "with no cap the step limit ends it, recorded as cut too")
    old = [0, 12, "A", ",".join(["continue"] + ["solver|blind"] * 11), ",".join(["0"] * 12)]
    check(P.cap_cut(loop, old), "an archived debate that ran out of decisions (before the fix) counts as cut")

    # 2. two caps (--total-cap, 2026-10-07): the plan may cost at most the turn cap (15); rules may add
    # rounds up to the total cap (21); a round that would pass 21 is not run and the cut is recorded
    check(P.RUN_CAP == P.MAX_TURNS == 15, "without --total-cap one cap bounds both (every earlier run)")
    s21 = P.configure_executor(executor="v3", window=32768, plain_instruction=True, last_round_vote=True,
                               count_read_summaries=True, high_cost=3, turn_cap=15, total_cap=21)
    check(P.MAX_TURNS == 15 and P.RUN_CAP == 21 and s21.get("total_cap") == 21 and s21.get("turn_cap") == 15,
          "--total-cap 21: the plan cap stays 15, the debate cap is 21, both in the settings")
    over = norm({"plan": [P.PLAN_ROUNDS["solver|high"], P.PLAN_ROUNDS["solver_x4|high"]],
                 "rules": [P.CONT(), {"when": ["step==2"], "do": "verifier"}], "default": "stop:last_commit"})
    out = M.run_program(over, agree, row, max_calls=P.RUN_CAP, grade=False)
    check(P.plan_cost(over) == 15 and P.never_runs_as_written(over) is None and out["n_calls"] == 16
          and out["actions"][-1] == "verifier" and not out["capped"],
          "a 15-turn plan is valid, and a rule's verifier after it runs (16 of 21 turns)", str(out["actions"]))
    out = M.run_program(loop, agree, row, max_calls=P.RUN_CAP, grade=False)
    check(out["n_calls"] == 21 and out["capped"], "a repeating rule runs to 21 turns and is recorded as cut")
    long_plan = norm({"plan": [P.PLAN_ROUNDS["solver_x4|high"], P.PLAN_ROUNDS["solver_x2|high"]],
                      "rules": [P.CONT()], "default": "stop:vote"})
    check(P.plan_cost(long_plan) == 18 and P.never_runs_as_written(long_plan) is not None,
          "a plan over 15 turns is refused (redrawn), even under a 21-turn debate cap")
    lit = [name for name in P.PROTOCOLS if P.never_runs_as_written(norm(P.PROTOCOLS[name])) is not None]
    check(not lit, "every literature program runs as written", str(lit))
    sr = norm(P.PROTOCOLS["self_refine_high"])
    flips = M.run_program(sr, Pattern(lambda r, j: "AB"[r % 2]), row, max_calls=P.RUN_CAP, grade=False)
    check(P.plan_cost(sr) == 15 and flips["n_calls"] == 15 and not flips["capped"],
          "self_refine_high is unchanged: 2 refine rounds, 15 turns, not cut")
    import argparse
    ap = argparse.ArgumentParser()
    P.add_executor_args(ap)
    ap.add_argument("--max-calls-per-question", type=int, default=None)
    args = ap.parse_args(["--context-window", "32768", "--high-cost", "3", "--turn-cap", "15", "--total-cap", "21"])
    P.configure_from_args(args)
    check(args.max_calls_per_question == 21, "the entry scripts' per-question cap is the debate cap (21)")
    try:
        P.configure_executor(executor="v3", window=32768, turn_cap=15, total_cap=12)
        check(False, "a total cap below the turn cap is refused")
    except SystemExit:
        check(True, "a total cap below the turn cap is refused")
    P.configure_executor(executor="v3", window=32768, plain_instruction=True, last_round_vote=True,
                         count_read_summaries=True, high_cost=3, turn_cap=15)

    # 3. duplicates (program_space.behaviour_hash): a program that answers alike at fewer turns is not a duplicate
    a = {"q1": [1, 3, "A", "continue", "0", 100]}
    b = {"q1": [1, 9, "A", "continue", "0", 300]}
    check(P.behaviour_hash(a) != P.behaviour_hash(b) and P.behaviour_hash(a) == P.behaviour_hash(dict(a)),
          "the turns are part of a program's identity")

    # 4. the tie floor (evolve_pipeline_cluster.settle_slots): a one-run score never sets it
    qs = [f"floor{i}" for i in range(10)]
    groups = {"groups": [{"group": 0, "search": qs, "held_out": []}], "per_group": 0, "k": 1}
    s = V.Search(types.SimpleNamespace(out=str(TMP / "floor"), seed=0, max_calls_per_question=15, tie_questions=1,
                                       workers=1), {}, groups, types.SimpleNamespace(), {})
    ones = lambda k: [1] * k + [0] * (10 - k)

    def add(name, width, marks, turns, second=None):
        p = norm({"plan": [P.PLAN_ROUNDS["solver" if width == 1 else f"solver_x{width}"]], "rules": [P.CONT()],
                  "default": "stop:vote"})
        r = P.ProgRecord(p, name, name, 0)
        for rep, mk in ((0, marks), (1, second)):
            if mk is not None:
                r.reps[rep] = {q: [mk[i], turns, "A", "continue", "0", 1] for i, q in enumerate(qs)}
        s.archive[r.key] = r
        s.mark_duplicate(r)
        return r
    cheap = add("cheap", 1, ones(6), 1.0, ones(6))
    add("mid", 2, ones(7), 2.0, ones(7))
    lucky = add("lucky", 3, ones(8), 9.0)                  # 0.8 on its one run
    asked = []

    def second_run(jobs, label):                           # the lucky program's second run: 0.4
        asked.extend(jobs)
        for rec, q, rep in jobs:
            rec.reps.setdefault(rep, {})[q] = [ones(4)[qs.index(q)], 9.0, "A", "continue", "0", 1]
        return 0
    s.run_jobs = second_run
    s.save = lambda rec: None
    s.settle_slots()
    check(any(r is lucky for r, _, _ in asked) and s.archive[s.slots[(0, "A")]] is cheap,
          "the program that sets the tie floor is run a second time; then the cheap holder keeps the slot",
          s.archive[s.slots[(0, "A")]].name)

    # 5. a torn last line (evolve_pipeline_cluster.end_line) does not swallow the next record
    torn = TMP / "torn.jsonl"
    torn.write_text('{"a": 1}\n{"b": ')
    V.end_line(torn)
    with torn.open("a") as fh:
        fh.write('{"c": 3}\n')
    got = []
    for line in torn.read_text().splitlines():
        try:
            got.append(json.loads(line))
        except ValueError:
            pass
    check(got == [{"a": 1}, {"c": 3}], "a record written after a torn line is kept", str(got))

    # 6. r1_majority counts the first round the program runs (program_space.opening_width)
    rule_first = norm({"plan": [P.PLAN_ROUNDS["solver"]],
                       "rules": [{"when": ["step==0"], "do": "solver_x4"}, P.CONT()], "default": "stop:vote"})
    check(P.opening_width(rule_first) == 4 and P.unsatisfiable("r1_majority>=3", rule_first) is None,
          "a program that opens with a rule's 4-solver round may ask r1_majority>=3")
    verifier_first = norm({"plan": [P.PLAN_ROUNDS["solver_x4"]],
                           "rules": [{"when": ["step==0"], "do": "verifier"}, P.CONT()], "default": "stop:vote"})
    check(P.unsatisfiable("r1_majority>=2", verifier_first) is not None,
          "one that opens with a 1-speaker rule round may not ask r1_majority>=2")

    # 7. a summary call that meets server trouble raises (debate_mcq._followup_v3): the round is run again
    import httpx
    import openai

    class Raising:
        def __init__(self, exc):
            self.exc = exc
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **kw):
            raise self.exc
    down = openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:1/v1/chat/completions"))
    try:
        D._followup_v3(Raising(down), QWEN, "sys", "q", "reply", "nudge", 100)
        check(False, "a summary call that cannot reach the server raises")
    except openai.APIConnectionError:
        check(True, "a summary call that cannot reach the server raises")
    check(D._followup_v3(Raising(ValueError("400: bad request")), QWEN, "sys", "q", "reply", "nudge", 100) == "",
          "a refused summary call still returns '' (part of the reply's record)")

    # 8. debates that keep failing (evolve_pipeline_cluster.run_jobs): wait for the server, or stop nonzero
    prog = norm({"plan": [P.PLAN_ROUNDS["solver"]], "rules": [P.CONT()], "default": "stop:vote"})
    rows = {f"j{i}": dict(row, id=f"j{i}") for i in range(3)}
    state = {"up": True, "fail": True, "polls": 0}

    class Runner:
        novel_calls = 0

        def reset_budget(self, b):
            pass

        def stage(self, s):
            pass

        def run_round(self, qid, rounds, specs, spec, rep=0, prompts=None):
            if state["fail"]:
                raise M.OffCache(qid)
            return [(p, "ANSWER: A") for p in spec["personas"]]
    shim = types.SimpleNamespace(runner=Runner(), rows=rows,
                                 args=types.SimpleNamespace(workers=2, max_calls_per_question=15, base_urls="x", model="m"))
    real_up, real_sleep = V.server_up, V.time.sleep

    def up(args):
        state["polls"] += 1
        if not state["up"] and state["polls"] >= 3:      # the server comes back, the debates then work
            state["up"], state["fail"] = True, False
        return state["up"]
    V.server_up, V.time.sleep = up, lambda s: None
    try:
        rec = P.ProgRecord(prog, "x", "x", 0)
        try:
            V.Search.run_jobs(shim, [(rec, q, 0) for q in rows], "always failing, server up")
            check(False, "debates failing with the server up stop the search nonzero")
        except SystemExit as exc:
            check("fail on every try" in str(exc) and not rec.reps.get(0),
                  "debates failing with the server up stop the search nonzero, nothing recorded", str(exc)[:80])
        state.update(up=False, fail=True, polls=0)
        rec = P.ProgRecord(prog, "y", "y", 0)
        V.Search.run_jobs(shim, [(rec, q, 0) for q in rows], "server down, then back")
        check(len(rec.reps.get(0, {})) == 3, "a server that goes down and comes back: the search waits, then completes")
    finally:
        V.server_up, V.time.sleep = real_up, real_sleep
    P.configure_executor(executor="v3", window=65536)


def test_reviewer_first():
    """No critic, verifier or synthesizer in a program's first round (2026-10-07): the search draws
    another edit, a seed with one is refused, the seed copier refuses it, and earlier archives still load."""
    print("no reviewer in the first round")
    P.configure_executor(**RUN2)
    R = P.PLAN_ROUNDS
    prog = lambda plan, rules=(): P.normalize_program({"plan": [R[x] for x in plan], "rules": [P.CONT(), *rules],
                                                       "default": "stop:vote"})
    first = lambda plan, act: P.normalize_program({"plan": [R[x] for x in plan],
                                                   "rules": [{"when": ["step==0"], "do": act}, P.CONT()],
                                                   "default": "stop:vote"})
    yes = [prog([x]) for x in ("verifier", "critic|high", "synthesizer", "verifier|high")]
    yes += [prog(["verifier", "solver_x2"]), first(["solver"], "critic|high")]
    no = [prog(["solver"]), prog(["expert_solver", "verifier"]), prog(["solver_x2", "critic", "synthesizer"]),
          first(["verifier"], "solver_x2"), P.normalize_program(P.PROTOCOLS["self_refine_high"])]
    check(all(P.reviewer_first(p) for p in yes) and not any(P.reviewer_first(p) for p in no),
          "a reviewer in the first plan round, or in the round a rule runs first, is caught; later ones are not")
    check(all(P.validate_program(p) is None for p in yes), "such programs are still valid (earlier archives load)")
    check(not any(P.reviewer_first(P.normalize_program(p)) for p in P.PROTOCOLS.values()),
          "no literature program opens with a reviewer")
    seeds = json.loads((P.ROOT / "outputs/pipeline_cluster_qwen9b/run2/seeds.json").read_text())["seeds"]
    check(len(seeds) == 11 and not any(P.reviewer_first(P.normalize_program(s["program"])) for s in seeds),
          "none of run2's 11 seeds opens with a reviewer")
    # the draw: children of parents one edit away from a reviewer first are never one
    s = V.Search(types.SimpleNamespace(out=str(TMP / "v3r2_rf"), seed=0, max_calls_per_question=21),
                 {}, P.load_groups(T3.CLUSTERS_V3, 3), None, {})
    s.replay = lambda rec, qids, rep=0: None                    # no cache: every child is new
    parents = [prog(["solver", "verifier"]), prog(["solver_x2|high", "critic", "solver"]),
               first(["verifier|high"], "solver_x2"), prog(["expert", "synthesizer"])]
    rng, stats, kids = random.Random(3), Counter(), []
    for i in range(400):
        par = P.ProgRecord(parents[i % 4], f"p{i}", f"p{i}", 0)
        kid = s.draw_child(par, 0, "A", f"k{i}", [], set(), rng, stats, donors=[prog(["verifier|high"])])
        if kid is not None:
            kids.append(kid)
    check(len(kids) > 300 and not any(P.reviewer_first(k.program) for k in kids) and stats["redraw_reviewer_first"] > 20,
          "400 draws from such parents: no child opens with a reviewer; the others are drawn again",
          f"{len(kids)} children, {stats['redraw_reviewer_first']} drawn again")
    # a seed with a reviewer first is refused (the copier and the search alike)
    src = {"settings": {}, "seeds": [{"name": "verifier_first", "source": "llm", "program": prog(["verifier|high"])}]}
    source, out = TMP / "v3r2_rf" / "src.json", TMP / "v3r2_rf" / "seeds.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(src))
    try:
        run_cli(CS, ["--source", str(source), "--out", str(out), "--context-window", "65536"] + RUN2_ARGS)
        refused = ""
    except SystemExit as exc:
        refused = str(exc)
    check("verifier_first: a critic, verifier or synthesizer speaks in its first round" in refused
          and not out.exists(), "the seed copier refuses a seed with a reviewer first", refused)
    try:
        s.seed([src["seeds"][0]])
        check(False, "the search refuses a seed with a reviewer first")
    except SystemExit as exc:
        check("first round" in str(exc), "the search refuses a seed with a reviewer first", str(exc))
    runner = (P.ROOT / "run_pipeline_cluster.sh").read_text()
    check(runner.count('--baselines ""') == 1 and "--baselines direct_high,self_refine_high" not in runner,
          "run2's champion step runs the finalists only")
    P.configure_executor(executor="v3", window=65536)


def test_edit_families():
    """The cluster search draws a family of edits evenly, then a kind evenly within it (2026-10-07);
    the global search keeps the even draw over kinds, unchanged."""
    print("edits drawn by family")
    P.configure_executor(**RUN2)
    R = P.PLAN_ROUNDS
    prog = P.normalize_program({"plan": [R["solver_x2"], R["solver"]],
                                "rules": [P.CONT(), {"when": ["step==2", "n_distinct>=2"], "do": "verifier"}],
                                "default": "stop:vote"})
    donor = P.normalize_program(P.PROTOCOLS["direct_high"])
    kinds = P.applicable_edits(prog, [donor])
    fams = {P.edit_family(k) for k in P.EDIT_KINDS}
    check(len(kinds) == 13 and fams == set(P.EDIT_FAMILIES) and len(fams) == 6
          and sorted(k for ks in P.EDIT_FAMILIES.values() for k in ks) == sorted(P.EDIT_KINDS + (P.WIDTH_OP,)),
          "6 families hold every one of the 13 kinds (and set_width) once; all 13 apply to the test program")

    def old_uniform(p, rng, donors):         # mutate_uniform as it was before the family draw
        ks = P.applicable_edits(p, donors)
        while ks:
            op = rng.choice(ks)
            for _ in range(20):
                if op in P.PLAN_OPS:
                    child, name = P.mutate_plan(p, rng, op)
                elif op in P.ROUND_OPS:
                    child, name = P.mutate_round(p, rng, op)
                elif op == P.WIDTH_OP:
                    child, name = P.mutate_width(p, rng)
                elif op == P.CROSSOVER:
                    child, name = P.crossover(p, rng.choice(list(donors)), rng), op
                else:
                    child, name = P.mutate_rules(p, rng, op, None)
                try:
                    P.validate_program(child)
                except (AssertionError, ValueError):
                    continue
                if P.canon(child) != P.canon(p):
                    return child, name
            ks.remove(op)
        return P.copy_program(p), "none"
    parents = [prog, donor, P.normalize_program(P.PROTOCOLS["self_refine_high"]),
               P.normalize_program(P.PROTOCOLS["early_exit_agree"])]
    rng_a, rng_b, same = random.Random(5), random.Random(5), True
    for i in range(800):
        a, b = P.mutate_uniform(parents[i % 4], rng_a, donors=[donor]), old_uniform(parents[i % 4], rng_b, [donor])
        same &= P.canon(a[0]) == P.canon(b[0]) and a[1] == b[1]
    check(same, "the global search's draw over kinds gives exactly the children it gave before")

    rng, got = random.Random(11), Counter()
    for _ in range(30000):
        got[P.mutate_by_family(prog, rng, donors=[donor])[1]] += 1
    fam = Counter()
    for k, v in got.items():
        fam[P.edit_family(k)] += v / 30000
    check(all(abs(v - 1 / 6) < 0.012 for v in fam.values()) and len(fam) == 6
          and all(abs(got[k] / 30000 - 1 / 36) < 0.006 for k in P.RULE_OPS)
          and all(abs(got[k] / 30000 - 1 / 18) < 0.008 for k in ("plan_replace", "plan_add", "plan_drop")),
          "all 6 families open: each family 1/6; each rule kind 1/36, each plan kind 1/18",
          str({k: round(v, 3) for k, v in fam.items()}))
    first = P.normalize_program({"plan": [R["solver_x2"]], "rules": [P.CONT()], "default": "stop:vote"})
    open_ = {P.edit_family(k) for k in P.applicable_edits(first, [])}
    rng, fam = random.Random(12), Counter()
    for _ in range(20000):
        fam[P.edit_family(P.mutate_by_family(first, rng)[1])] += 1 / 20000
    check(open_ == set(fam) and len(fam) < 6 and all(abs(v - 1 / len(fam)) < 0.015 for v in fam.values()),
          "fewer families open: each open family equally likely", f"{sorted(open_)} {dict(fam)}")

    # which draw each search calls
    calls = Counter()
    real = P.mutate_uniform, P.mutate_by_family
    P.mutate_uniform = lambda *a, **k: (calls.update(["uniform"]), real[0](*a, **k))[1]
    P.mutate_by_family = lambda *a, **k: (calls.update(["family"]), real[1](*a, **k))[1]
    try:
        import evolve_pipeline_global as V4
        for name, search in (("cluster", V.Search), ("global", V4.Search)):
            s = V.Search(types.SimpleNamespace(out=str(TMP / f"v3r2_fam_{name}"), seed=0, max_calls_per_question=21),
                         {}, P.load_groups(T3.CLUSTERS_V3, 3), None, {})
            s.replay = lambda rec, qids, rep=0: None
            before = Counter(calls)
            for i in range(30):
                search.draw_child(s, P.ProgRecord(prog, f"p{i}", f"p{i}", 0), 0, "A", f"k{i}", [], set(),
                                  random.Random(i), Counter(), donors=[donor])
            calls_now = calls - before
            check(set(calls_now) == {"family" if name == "cluster" else "uniform"},
                  f"the {name} search draws {'by family' if name == 'cluster' else 'over kinds'}", str(dict(calls_now)))
    finally:
        P.mutate_uniform, P.mutate_by_family = real
    P.configure_executor(executor="v3", window=65536)


def test_stopped_generation():
    """A search stopped part way through a generation (2026-10-07): on resume that generation's children
    are dropped (the archive as it was is kept) and it is drawn and run again from its start."""
    print("a generation stopped part way")
    seeds_path = TMP / "v3r2" / "seeds.json"                # written by test_search_run2 (fits run2's caps)
    run, cache = TMP / "v3r2_stop" / "run", TMP / "v3r2_stop" / "rounds.jsonl"
    common = ["--seeds", str(seeds_path), "--out", str(run), "--clusters", str(T3.CLUSTERS_V3), "--per-group", "3",
              "--live-cache", str(cache), "--workers", "8", "--tie-questions", "1"] + T3.WINDOW + RUN2_ARGS
    run_cli(V, common + ["--generations", "2"])
    real = V.Search.run_jobs

    def stop_in_wave_2(self, jobs, label):
        if label.startswith("gen 3 wave 2"):
            raise SystemExit("simulated stop")
        return real(self, jobs, label)
    V.Search.run_jobs = stop_in_wave_2
    try:
        run_cli(V, common + ["--generations", "3", "--resume"])
        check(False, "the simulated stop happened")
    except SystemExit as exc:
        check(str(exc) == "simulated stop", "the search stops in generation 3, wave 2", str(exc))
    finally:
        V.Search.run_jobs = real
    lines = lambda p: [json.loads(l) for l in p.open() if l.strip()]
    before = {d["key"]: d for d in lines(run / "archive.jsonl") if not d.get("header")}
    stopped = {k for k, d in before.items() if d["gen"] == 3}
    n_slots = len(lines(run / "generations.jsonl")[-1]["slots"])
    check(0 < len(stopped) <= n_slots and max(d["gen"] for d in lines(run / "generations.jsonl")) == 2,
          "the stop left generation 3's first wave (one child per slot at most) in the archive, unlogged",
          f"{len(stopped)} children, {n_slots} slots")
    run_cli(V, common + ["--generations", "3", "--resume"])
    after = {d["key"]: d for d in lines(run / "archive.jsonl") if not d.get("header")}
    gens = lines(run / "generations.jsonl")
    gen3 = {k for k, d in after.items() if d["gen"] == 3}
    backup = run / "archive.jsonl.stopped_gen3"
    check(backup.exists() and {d.get("key") for d in lines(backup) if not d.get("header")} >= stopped,
          "the archive as it was is kept beside it")
    check(gens[-1]["gen"] == 3 and gens[-1]["children"] == len(gen3) and len(gen3) <= 2 * n_slots
          and all(d["gen"] < 3 or k in gen3 for k, d in after.items())
          and {k: d for k, d in after.items() if d["gen"] < 3}.keys() == {k for k, d in before.items() if d["gen"] < 3},
          "generation 3 is drawn again: the archive holds only the children the log counts, the earlier ones kept",
          f"logged {gens[-1]['children']}, archived {len(gen3)}")
    names = [d["name"] for d in after.values() if d["gen"] == 3]
    check(len(names) == len(set(names)) and not any(n.endswith("r") for n in names),
          "the redrawn children take the generation's own names", str(sorted(names)))


if __name__ == "__main__":
    P.configure_executor(executor="v3", window=65536)      # the entry scripts do the same
    test_audit_fixes()
    test_prompts()
    test_every_request()
    test_discussion()
    test_read_summaries()
    test_cap_record()
    test_tie_rule()
    test_self_refine_stop()
    test_against_external()
    T3.test_subsets()
    seeds_path = T3.test_seeds()
    test_copy_seeds(seeds_path)
    test_reviewer_first()                                  # after test_subsets: it reads the fake groups
    test_edit_families()
    P.configure_executor(executor="v3", window=65536)
    test_search_run2(seeds_path)
    test_stopped_generation()
    print()
    if T.FAILURES:
        print(f"{len(T.FAILURES)} FAILED: " + "; ".join(T.FAILURES))
        sys.exit(1)
    print(f"all checks passed (model calls: {T.MODEL_CALLS}); temp dir {TMP}")

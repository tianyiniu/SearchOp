#!/usr/bin/env bash
# The whole cluster-pipeline experiment for one model:
#   1. external baselines on the test split: direct chain-of-thought and Self-Refine, 3 runs each
#   2. the same external baselines on the train search questions (and only on them)
#   3. seeds on the train search questions
#   4. search, generation 0: every seed on every search question, two replicates; then a check
#      that every seed ran a round on every question (the run stops here if one did not)
#   5. the seeds against the external baselines of step 2, on the search questions
#   6. search, generations 1..5
#   7. a look at the test split halfway: the slot holders of generation 5 (and the rows of step 9),
#      1 replicate -> <run>/test_eval_gen5
#   8. search, generations 6..10
#   8b. only with --champions and a dev split: the champion step on the dev questions -> <run>/champions.json
#   9. evaluation on the test split, 3 replicates -> <run>/test_eval: the routed slot-A holders (each
#      group's strongest grid program) and the global slot holder; with --champions and a dev split,
#      the routed champions (step 8b), beside them the routed slot-A holders, and the global champion;
#      then the external baselines of step 1. Each group's program runs only on its own group's
#      questions, the global program on every question (eval_routed_dev.py --routed-only), and no
#      in-executor protocols run (since 2026-10-09)
# Every step is resumable: run the script again after an interruption and finished work is kept.
#
#     bash run_pipeline_cluster.sh <gptoss|qwen|qwen9b> [supergpqa|hle] [--champions]   (the dataset defaults to supergpqa)
#
#     --champions: run step 8b and evaluate its champions in step 9. Off by default (2026-10-09): after the
#     search, step 9 runs the strongest grid programs, and the dev questions (if any) are not used. The
#     dev split itself is unchanged, so the search questions are the same either way.
#
#     mkdir -p outputs/pipeline_cluster_gptoss/run4
#     bash run_compare_external_cluster.sh gptoss 2>&1 | tee -a outputs/pipeline_cluster_gptoss/run4/compare.log
#     bash run_pipeline_cluster.sh gptoss 2>&1 | tee -a outputs/pipeline_cluster_gptoss/run4/pipeline.log
#     mkdir -p outputs/pipeline_cluster_hle_gptoss/run1
#     bash run_pipeline_cluster.sh gptoss hle 2>&1 | tee -a outputs/pipeline_cluster_hle_gptoss/run1/pipeline.log
#     (qwen: outputs/pipeline_cluster_qwen35b, outputs/pipeline_cluster_hle_qwen35b)
#     mkdir -p outputs/pipeline_cluster_qwen9b/run2
#     bash run_pipeline_cluster.sh qwen9b 2>&1 | tee -a outputs/pipeline_cluster_qwen9b/run2/pipeline.log
#
# gptoss SuperGPQA now runs <out>/run4 (2026-10-07): qwen9b run2's design, settings and prompts (below;
# prompts' signature 032e5eb669 for both models), and qwen9b run2's seeds (copied, step 3; the seed
# writer's text does not depend on the debate model). It drops --visible-reasoning, which runs 1-3 used:
# since 2026-10-06 every speaker is asked to write out its reasoning, and the summary later speakers read
# of a reply whose reasoning is hidden (gpt-oss thinks in a hidden channel) is made from that reasoning. Its comparison with the external baselines (run_compare_external_cluster.sh
# gptoss) must pass first, as for qwen9b. gptoss HLE (run1) keeps its earlier settings.
#
# qwen9b now runs <out>/run2 (2026-10-06). It differs from qwen9b run1 (below) in these ways:
#   - the prompts (rewritten 2026-10-06 for every run, debate_mcq): a solver's system prompt is a general
#     one and its user message is SuperGPQA's own zero-shot prompt, the external direct baseline's
#     request (in run1 a high-effort solver used about 1.6 times that baseline's tokens and scored about
#     2 points lower); the critic reads the whole discussion, not only the leading answer; no prompt
#     says "commit"; a speaker that sees the discussion sees it round by round (how many speakers each
#     round had and what they saw) and is told which round it speaks in;
#   - --count-read-summaries: a summary is still written for every long reply, but a debate's tokens
#     leave out the summaries that no later speaker read (they never change an answer);
#   - the budget: a high-effort speaker counts 3 turns (was 5); a program's plan (its opening rounds)
#     may cost at most 15 turns (was 16), so at most 5 high-effort plan speakers (was 3); rules may add
#     rounds up to 21 turns per question, and a round that would pass 21 is not run (the default stop
#     decides; the cut is recorded) (--high-cost 3 --turn-cap 15 --total-cap 21, 2026-10-07);
#   - no critic, verifier or synthesizer in a program's first round (every cluster-pipeline run from
#     2026-10-07; program_space.reviewer_first): they review the answers given so far, and there are
#     none yet. The search draws another edit instead, a seed with one is refused, and the seed writer
#     is told. Earlier archives still load;
#   - edits are drawn by family (every cluster-pipeline run from 2026-10-07; program_space.
#     mutate_by_family): a family evenly among rule, plan, width, effort, visibility and crossover,
#     then a kind evenly within it. Every earlier run drew the 13 kinds evenly, so the 6 rule kinds
#     made 37% of the children; over five runs 7% of their children beat their parent, against 12%
#     for the other edits. By family, rule edits make about 16%;
#   - step 9's in-executor rows are direct_high and self_refine_high only (2026-10-07; the matches of the
#     external Direct CoT and Self-Refine, which think): the low-effort direct, self_refine and mad match
#     no reported baseline. Nothing on the test split is recorded with these prompts, so all run anew;
#   - the champion step chooses among the finalists only and runs no baselines beside them
#     (direct_high and self_refine_high were shown there but could never be chosen; the test split
#     compares with them);
#   - --tie-questions 1 (the search and the champion step): every program within one question of a
#     slot's best score ties with it, and the fewest turns wins (run1 needed an exactly equal score);
#   - before it, a separate script, run_compare_external_cluster.sh qwen9b, compares our Direct CoT
#     (direct_high) and our Self-Refine (self_refine_high) with the external Direct CoT and Self-Refine on
#     the 200 search questions (accuracy, tokens, replies out of room, no letter) and ends; this script
#     refuses to start the search without a passed comparison (<run>/external_baselines.json);
#   - the shared setup (families, paths, options, steps 1 and 2) is in pipeline_cluster_setup.sh, which both
#     scripts source;
#   - self_refine_high is run as the external Self-Refine runs: the answer, then up to 2 feedback ->
#     refine rounds (a critic, then a solver), stopping after a critic that keeps the answer, as the
#     external one stops when its feedback says "it is correct" (program_space, 2026-10-06);
#   - the seeds are written anew (step 3, 2026-10-07): run1's model-written seeds were written while
#     the seed writer saw example questions that are run2 dev questions (run1 had no dev split); now it
#     sees search questions only. The train baselines start from copies of run1's whole-train files
#     (they hold the 200 search questions), so they do not run again;
#   - no look at the test split during the search (step 7 is skipped);
#   - the dev split is used (100 held-out questions, the champion step): in run1 it was set to 0 by
#     mistake, so run1 had no champions;
#   - the fixes of the 2026-10-06 audit (all runs; none changes a prompt or a recording's key):
#     a debate is ended by the turn cap alone (a 12-decision limit used to cut loops of cheap rounds
#     below the cap, unrecorded: 635 debates of run1); a program that answers like another at fewer turns is no
#     longer a duplicate of it; the program that sets a slot's tie floor is run a second time too;
#     debates that fail are retried while the server is down (up to 2 h) or slowly while it answers,
#     and then the search stops nonzero instead of archiving programs with debates missing (a resume
#     fills any that are); a summary or commit call that meets a server error makes its round run
#     again instead of being kept without its summary or letter; step 9 stops nonzero while test
#     debates are missing; the champion step counts missing debates of both replicates; a torn last
#     line no longer swallows the next record; the comparison gate also checks the settings and the
#     prompts' signature.
#   - To stop a running search, kill its python process (kill -9); Ctrl-C can hang it. A resume after
#     kill -9 keeps all finished generations. A generation stopped part way is drawn again from its
#     start: its children are dropped (the archive as it was is kept as archive.jsonl.stopped_gen<n>)
#     and new ones are drawn and run.
#
# qwen9b run1 (Qwen3.5-9B, SuperGPQA, <out>/run1; 2026-10-04) differed from run3 in five ways:
#   - run2's budget: a high-effort speaker counts 5 turns, a question may use 16 (the defaults), no
#     judge speaker, the width edit on the first plan round only; run2's programs were the cheapest
#     good ones (gpt-oss test: 48.3% avg@3 at 11.8k tokens against 42k in run3);
#   - --plain-instruction: every speaker is asked only for the ANSWER line, without the request to
#     think carefully and take the space it needs; the effort sent with the request alone sets how
#     long it thinks (on the global-pipeline Qwen run that request made a high-effort solver think 1.5x longer and
#     run out of the window on 3.6% of calls, at no gain in accuracy);
#   - --last-round-vote: the stops last_commit and last_speaker read the most common answer of the
#     last round that committed one (the global pipeline's read repair);
#   - a dev split: 100 of the 300 train questions (split_train_dev.py, seed 0: the global-pipeline runs' split)
#     are held out; the search runs on the other 200, and after generation 10 the champion step
#     (--pick-champions) runs the 5 strongest distinct programs of each group, and of all groups,
#     on that group's dev questions (2 replicates) and keeps the best per group; step 9 routes the
#     test questions to those champions (and still shows the strongest grid programs beside them);
#   - nothing is copied from another run (nothing on record has these prompts).
#
# HLE (the text-only splits, datasets/hle_text_{train_800,test_200}.json; groups from
# run_describe_hle.sh in outputs/describe_hle_v4, k = 4) runs with open answers: a speaker ends with
# 'ANSWER: <answer>' in a fixed format (the letter alone for a question with answer choices, else
# the answer alone in its simplest exact form), answers are compared after normalisation (formatting
# removed, case kept) for votes and agreement, and a final answer is graded by a judge model
# (gpt-6-luna, HLE's judge prompt; scripts/judge_answers.py). The external baselines get HLE's own
# response format and the same judge. Every verdict is cached once in <out>/judge_gpt-6-luna.jsonl
# and shared by the search, the test evaluation and the baselines.
#
# SuperGPQA: the search goes to <out>/run3 (2026-09-30), with three changes from run2:
#   - every train question is a search question (PER_GROUP=0: 144 + 99 + 57 = 300, not 50 per group);
#   - the budget: a high-effort speaker counts 3 turns (was 5) and a question may use 20 (was 16),
#     which is also the most one round may cost (--high-cost 3 --turn-cap 20);
#   - the judge speaker (--judge-persona): it sees the debate and the list of answers committed so
#     far, and must choose one of them (debate_mcq, "the judge speaker"); it is a plan round and a
#     move (judge, judge|high), and two seeds use it (program_space.judge_seeds);
#   - the width edit changes the number of solvers of any solver round, not only the first plan
#     round: a later plan round or an extra round run by a rule too (--any-round-width: set_width
#     in place of plan_width).
# run1 and run2 are kept for reference. The search's round cache (<out>/rounds_*.jsonl) is shared by
# all runs (none of the three changes is in a recording's key, so run2's debates on its 150 search
# questions replay for free); run3's external train baselines start from copies of run2's files, so
# only the 150 questions new to the search are run; the test steps start from a copy of run2's test
# debates (which hold run1's), so the in-executor protocols are not run again.
# HLE is unchanged: 50 search questions per group, the old budget, no judge (its run1 started under
# this script before the SuperGPQA changes, and a rerun resumes it with the same settings).
#
# Run it ON THE SERVER THAT HOSTS the model (bound to localhost), started at its 32,768-token
# window (Model_hosting/deploy_gpt_oss_20b.sh, deploy_qwen35_35b_fp8.sh). The searched programs'
# replies take their limit from the window the server reports (window - prompt - 2,048 kept for
# the summary), so nothing in them is set by hand.
#
# Data (scripts/sample_supergpqa_subsets.py, run_describe_v3.sh):
#   train  datasets/supergpqa_600_train.json, 300 questions in the groups of
#          outputs/describe_v3/clusters_600_train.json (144 / 99 / 57). Since run3 every question is a
#          search question (PER_GROUP=0), in the clusters file's representative order (the group's
#          centre, then alternating far and random picks), so run2's 50 per group come first. With
#          PER_GROUP=N each group gives at most N (a smaller group is used whole) and the rest are not
#          used: there is no champion step. The search questions are written to
#          <out>/search_questions_all.json (_capN with a cap; HLE: <out>/search_questions.json).
#   test   datasets/supergpqa_600_test.json, 300 questions, routed to those groups by
#          outputs/describe_v3/routes_600_test.json
#
# External baselines (baselines/generate.py, selfrefine.py, recover.py, score.py): the official
# SuperGPQA zero-shot prompt; gpt-oss at reasoning effort high, Qwen with thinking on and its model
# card's sampling. Token limits fit the 32,768-token window: direct 28,672 (the longest prompt of the
# 600 splits is 3,002 tokens); Self-Refine 24,576 per answer turn and 16,384 per feedback turn. At
# this window gpt-oss runs out of room before answering on 14.6% of its direct test samples, which
# the method as published scores as wrong. The reported baselines ("_rec" files) instead ask such a
# reply, once, for the letter its reasoning supports, as the debate executor does (recover.py):
# direct samples keep their reasoning, so they are recovered after generation; Self-Refine does it
# inside its loop (--recover) and falls back to its latest answer that named a letter. The files
# without "_rec" (the method exactly as published) are scored and shown too where they exist.
# Failed requests are retried by running the scripts again, up to 3 times.
# Results go to baselines/results/{direct,selfrefine}_<model tag>_<question set>[_rec][_k3|_k1].json[l].
#
# The searched programs (scripts/evolve_pipeline_cluster.py and debate_mcq.chat_v3 have the
# details): every round is low or high effort (sent on every request) and sees the debate or only
# the question; no reply cap (a reply may use the window less a reserve for its summary);
# model-card sampling per effort; a committed reply of <= 500 words is shown to later speakers as
# it is, a longer one summarised with its letter locked; blind speakers are cached one by one and
# shared across programs. A program's rules are checked before the first round and after every
# round; a program that would stop before its first round is invalid (never seeded or bred). The
# search keeps an A and a B slot per group and one global slot, prunes holders of rules that never
# fired before they breed, and edits by an even draw over the edit families (rule, plan, width,
# effort, visibility, crossover; every edit kind evenly before 2026-10-07). Every child is run once on every search question; a program that would
# take a slot is run a second time on that slot's questions, and keeps the slot only if the two-run
# average still wins. Each generation breeds 2 children per slot, 14 in all.
#
# Seeds: one model-written program per group (program_guide.GUIDE_MODEL, gpt-6-sol) and 8 literature
# programs: the high-effort direct_high, self_refine_high, self_consistency_high,
# verify_then_decide_high, fresh_on_disagree_high, and mad, early_exit_agree, expert_first (which
# have no high-effort version; their low-effort originals are not seeds); with the judge also the two
# judge seeds, judge_on_disagree_high and pool_judge_high. 3 + 8 + 2 = 13 at k = 3 (SuperGPQA run3),
# 4 + 8 = 12 at k = 4 (HLE). Step 3 makes them (the model-written ones are written anew for run3,
# with the judge and the new budget in the grammar they are shown), unless <run>/seeds.json exists.
# The seed writer sees each group's profile (medoid template, typical steps, failure risks,
# knowledge) and 10 of its questions with their options, no answers (program_seeds_cluster
# --seed-examples); no difficulty labels and no scores. The archive only grows, so no group's A-slot holder can
# score below the best seed on that group's search questions, nor the global holder below the best
# seed on all of them (with --tie-questions 1: by more than one question, in exchange for fewer turns).
set -euo pipefail
cd "$(dirname "$0")"
PICK_CHAMPIONS=0                                     # --champions: step 8b and its champions in step 9
ARGS=()
for a in "$@"; do
    if [[ "$a" == --champions ]]; then PICK_CHAMPIONS=1; else ARGS+=("$a"); fi
done
set -- ${ARGS[@]+"${ARGS[@]}"}                        # the family and the dataset, for the setup
source pipeline_cluster_setup.sh

# --- before the search: the comparison with the external baselines must have passed ----------------
if (( COMPARE_EXTERNAL )); then
    # the verdict, and the comparison's settings against this run's (the prompts' signature among them):
    # a pass made with other prompts or options says nothing about this run
    VERDICT="$("$PYTHON" - "$RUN/external_baselines.json" "$MODEL" "$WINDOW" "${EXECUTOR[@]}" <<'EOF'
import argparse, json, sys
sys.path.insert(0, "scripts")
import program_space as P
path, model, window, flags = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
try:
    d = json.load(open(path))
except (OSError, ValueError):
    print("none"); sys.exit(0)
ap = argparse.ArgumentParser()
P.add_executor_args(ap)
settings = P.configure_from_args(ap.parse_args(["--context-window", window] + flags))
settings["model"] = model
old = d.get("settings") or {}
if d.get("verdict") != "pass":
    print(d.get("verdict", "none"))
elif old != settings:
    print("made with other settings: " + ", ".join(sorted(k for k in set(old) | set(settings) if old.get(k) != settings.get(k))))
else:
    print("pass")
EOF
)"
    if [[ "$VERDICT" != pass ]]; then
        echo "no passed comparison of our Direct CoT and Self-Refine with the external ones under this run's settings" \
             "($RUN/external_baselines.json: $VERDICT): run bash run_compare_external_cluster.sh $FAMILY first" >&2
        exit 1
    fi
    echo "[$(stamp)] the comparison with the external baselines passed ($RUN/external_baselines.md)"
fi

# --- 3. seeds ------------------------------------------------------------------------------------
if [[ -n "$REUSE_SEEDS_FROM" && ! -f "$SEEDS" && -f "$OUT/$REUSE_SEEDS_FROM/seeds.json" ]]; then
    # the same seeds, with each literature seed as the code now defines it (scripts/copy_seeds_cluster.py)
    echo "[$(stamp)] step 3: $SEEDS starts from $OUT/$REUSE_SEEDS_FROM/seeds.json"
    "$PYTHON" scripts/copy_seeds_cluster.py --source "$OUT/$REUSE_SEEDS_FROM/seeds.json" --out "$SEEDS" \
        --model "$MODEL" --base-urls "$BASE_URL" --context-window "$WINDOW" "${EXECUTOR[@]}"
fi
if [[ -f "$SEEDS" ]]; then
    echo "[$(stamp)] step 3: $SEEDS exists, skipping"
else
    echo "[$(stamp)] step 3: seed stage"
    "$PYTHON" scripts/program_seeds_cluster.py --out "$SEEDS" "${COMMON[@]}" 2>&1 | tee -a "$RUN/seeds.log"
fi

# --- 4. search, generation 0 (a finished generation resumes without spending) -------------------
search() {  # generations -> run or resume the search up to that many generations
    local resume=""
    [[ -f "$RUN/archive.jsonl" ]] && resume="--resume"
    "$PYTHON" scripts/evolve_pipeline_cluster.py --seeds "$SEEDS" --out "$RUN" \
        "${COMMON[@]}" "${TIE[@]}" --generations "$1" $resume 2>&1 | tee -a "$RUN/search.log"
}
last_generation() {  # the last generation the search finished (-1 if none; a torn line is skipped)
    "$PYTHON" - "$RUN/generations.jsonl" <<'EOF'
import json, sys
gens = []
try:
    for line in open(sys.argv[1]):
        try:
            gens.append(int(json.loads(line)["gen"]))
        except (ValueError, KeyError, TypeError):
            pass
except FileNotFoundError:
    pass
print(max(gens or [-1]))
EOF
}
check_generation() {  # n -> stop unless the search has finished generation n (it stops nonzero on lasting errors)
    local got
    got="$(last_generation)"
    if (( got < $1 )); then
        echo "the search stopped at generation $got, not $1 (see $RUN/search.log); run the script again to resume" >&2
        exit 1
    fi
}
check_seeds() {  # every seed scored twice on every search question, and every one of those debates ran a round
    "$PYTHON" - "$SEEDS" "$RUN/archive.jsonl" <<'EOF'
import json, sys
seeds = json.load(open(sys.argv[1]))["seeds"]
lines = []
for l in open(sys.argv[2]):
    try:
        lines.append(json.loads(l))
    except ValueError:                               # a torn line (a kill during a write)
        pass
qids, recs = set(lines[0]["qids"]), {}
for d in lines[1:]:
    recs[d["key"]] = d                               # a program's last line is its current state
by_name = {d["name"]: d for d in recs.values() if d["gen"] == 0}
bad = []
for s in seeds:
    d = by_name.get(s["name"])
    if d is None:
        bad.append(f"{s['name']}: not in the archive")
        continue
    for rep in ("0", "1"):
        got = d["reps"].get(rep, {})
        if set(got) != qids:
            bad.append(f"{s['name']}: replicate {rep} covers {len(set(got) & qids)} of {len(qids)} questions")
        elif (dead := sum(v[1] < 1 for v in got.values())):
            bad.append(f"{s['name']}: replicate {rep} ran no round on {dead} of {len(qids)} questions")
if bad:
    print("generation 0 check FAILED (the search is stopped before generation 1):\n  " + "\n  ".join(bad))
    sys.exit(1)
src = {}
for s in seeds:
    src[s["source"]] = src.get(s["source"], 0) + 1
print(f"generation 0 check: all {len(seeds)} seeds {src} ran a round on all {len(qids)} search questions, twice")
EOF
}
echo "[$(stamp)] step 4: search, generation 0 (the seeds on every search question)"
search 0
check_generation 0
check_seeds

# --- 5. seeds vs external baselines on the search questions -----------------------------------
echo "[$(stamp)] step 5: seeds vs external baselines on the search questions -> $RUN/train_baselines.md"
"$PYTHON" scripts/train_baselines.py --compare --run "$RUN" --external "$(external "$BNAME_TRAIN")"

# --- 6. search, generations 1..MID --------------------------------------------------------------
echo "[$(stamp)] step 6: search, generations 1..$MID"
search "$MID"
check_generation "$MID"

# --- the test split: the programs of a finished search generation, on one round cache ------------
TEST_CACHE="$RUN/test_eval/rounds_$TAGNAME.jsonl"
test_eval() {  # out reps -> routed slot-A holders, global holder, in-executor protocols, external baselines
    local out="$1" reps="$2" old="$REUSE_TEST_FROM/rounds_$TAGNAME.jsonl"
    mkdir -p "$out" "$(dirname "$TEST_CACHE")"
    if [[ -n "$REUSE_TEST_FROM" && ! -f "$TEST_CACHE" && -f "$old" ]]; then
        echo "[$(stamp)]   the test debates start from a copy of $REUSE_TEST_FROM's, so the protocols it ran are reused"
        cp "$old" "$TEST_CACHE"
    fi
    local champs=(--no-champions)
    (( N_DEV > 0 && PICK_CHAMPIONS )) && [[ "$out" == "$RUN/test_eval" ]] && champs=()    # only after step 8b
    "$PYTHON" scripts/eval_routed_dev.py --run "$RUN" --routes "$ROUTES" --dataset "$TEST" "${champs[@]}" \
        --out "$out" --live-cache "$TEST_CACHE" --model "$MODEL" --base-urls "$BASE_URL" --workers "$WORKERS" \
        --reps "$reps" --baselines "$PROTOCOLS" --routed-only "${EXECUTOR[@]}" --external "$(external "$BNAME_TEST")" \
        2>&1 | tee -a "$out.log"
}

# --- 7. halfway: the slot holders of generation MID on the test split, 1 replicate -------------------
# It reads <run>/summary.json, so it runs only while the search stands at generation MID; once
# written, it is never redone (a rerun after generation MID would otherwise evaluate later holders).
MID_OUT="$RUN/test_eval_gen$MID"
if (( ! MID_TEST )); then
    echo "[$(stamp)] step 7: skipped for this run (no look at the test split during the search)"
elif [[ -f "$MID_OUT/results_k1.json" ]]; then
    echo "[$(stamp)] step 7: $MID_OUT/results_k1.json exists, skipping"
elif (( $(last_generation) != MID )); then
    echo "[$(stamp)] step 7: skipped: the search is already past generation $MID, so its slot holders are gone"
else
    echo "[$(stamp)] step 7: the slot holders of generation $MID on $TEST, 1 replicate -> $MID_OUT"
    test_eval "$MID_OUT" 1
fi

# --- 8. search, generations MID+1..GENERATIONS -------------------------------------------------------
echo "[$(stamp)] step 8: search, generations $((MID + 1))..$GENERATIONS"
search "$GENERATIONS"
check_generation "$GENERATIONS"

# --- 8b. with --champions and a dev split: the champions, chosen on the dev questions -------------
CHAMPIONS="$RUN/champions.json"
check_champions() {  # every finalist and baseline of every group ran on every dev question, twice
    "$PYTHON" - "$CHAMPIONS" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print("no complete champions file"); sys.exit(1)
bad = [f"{where}: {p['name']} misses {p['n_missing']}" for where, r in [*d["per_group"].items(), ("all", d["global"])]
       for p in r["programs"] if p["n_missing"]]
bad += [f"{where}: no champion" for where, r in [*d["per_group"].items(), ("all", d["global"])] if not r["champion"]]
if bad:
    print("champion step incomplete (server errors?):\n  " + "\n  ".join(bad)); sys.exit(1)
print("champion check: every program ran on every dev question")
EOF
}
if (( N_DEV > 0 && PICK_CHAMPIONS )); then
    if check_champions >/dev/null 2>&1; then
        echo "[$(stamp)] step 8b: $CHAMPIONS exists and is complete, skipping"
    else
        echo "[$(stamp)] step 8b: the champions per group on the dev questions -> $CHAMPIONS"
        rm -f "$CHAMPIONS"
        "$PYTHON" scripts/evolve_pipeline_cluster.py --seeds "$SEEDS" --out "$RUN" "${COMMON[@]}" "${TIE[@]}" \
            --pick-champions --reps 2 --baselines "" 2>&1 | tee -a "$RUN/search.log"   # finalists only
        check_champions || { echo "run the script again to redo the champion step" >&2; exit 1; }
    fi
elif (( N_DEV > 0 )); then
    echo "[$(stamp)] step 8b: skipped (no --champions): step 9 runs the strongest grid programs; the dev questions are not used"
fi

# --- 9. evaluation on the test split -------------------------------------------------------------
echo "[$(stamp)] step 9: routed programs and baselines on $TEST, $K replicates"
test_eval "$RUN/test_eval" "$K"

echo "[$(stamp)] done: $RUN/train_baselines.md, $( (( MID_TEST )) && echo "$MID_OUT/results_k1.md, ")$RUN/test_eval/results_k$K.md, $RUN/summary.json$( (( N_DEV > 0 && PICK_CHAMPIONS )) && echo ", $CHAMPIONS")"

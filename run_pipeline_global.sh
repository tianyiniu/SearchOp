#!/usr/bin/env bash
# The global-pipeline experiment for one model, on SuperGPQA (pipeline_global_plan.md has the reasons):
#   1. external baselines on the test split (direct and Self-Refine, 3 runs each; the cluster-pipeline runs
#      already made them, so normally this only scores them again)
#   2. the splits: the 300 train questions -> a train split of 200 (the search) and a dev split
#      of 100 (the final choice only), a fixed random draw (scripts/split_train_dev.py)
#   3. seeds: the 8 literature programs and one model-written program per cost level
#      (scripts/program_seeds_global.py)
#   4. search, generation 0: every seed once on every train question; then a check that every
#      seed ran a round on every one (the run stops here if one did not)
#   5. search, generations 1..10: one search over all train questions (no question groups, no
#      routing); the best program of each cost level breeds 5 children per generation
#      (scripts/evolve_pipeline_global.py)
#   6. the final choice on the dev split: per level, the 2 best programs on the train split and
#      the level's best seed, run once on the dev split; the best one is the level's final
#      program -> <run>/final.json
#   7. evaluation on the test split: the final programs, the in-executor direct_high and
#      self_refine_high, and the external baselines of step 1, 3 replicates -> <run>/test_eval
# Every step is resumable: run the script again after an interruption and finished work is kept.
# A step that cannot finish (the server down, a debate failing on every try) stops the script
# with exit code 1 and a message; run it again once the cause is fixed.
#
# Start it inside tmux (so it keeps going if the SSH session closes):
#
#     mkdir -p outputs/pipeline_global_gptoss/run1
#     bash run_pipeline_global.sh gptoss 2>&1 | tee -a outputs/pipeline_global_gptoss/run1/pipeline.log
#
# Executor settings, the same in every step: run3's (a high-effort speaker counts 3 turns, the
# width edit applies to any solver round) with a question capped at 15 turns, not 20 (in run3,
# turns past 15 bought nothing: -0.2 points on train, no change on test), with no judge speaker, and
# the final-read repair (--last-round-vote): the stops last_commit and last_speaker read the
# most common answer of the last round that committed one, not the last speaker's. None of
# these is in a recording's key, so the round caches start from copies of the cluster pipeline's: every
# debate run3 recorded on a train or test question replays for free.
#
# Cost levels, by average speaker turns per train question: cheap up to 4.5, medium up to
# 10.5, expensive above. Seeds and children join the level of their own cost.
#
# Run it ON THE SERVER THAT HOSTS the model (bound to localhost), started at its 32,768-token
# window (Model_hosting/deploy_gpt_oss_20b.sh, deploy_qwen35_35b_fp8.sh). The seed stage asks
# the guide model (gpt-6-sol), so OPENAI_API_KEY must be in .env.
set -euo pipefail
cd "$(dirname "$0")"

FAMILY="${1:?gptoss, qwen or qwen9b}"
case "$FAMILY" in
    gptoss)
        MODEL="openai/gpt-oss-20b"; BASE_URL="${BASE_URL:-http://localhost:7472/v1}"
        FTAG="gptoss"; WORKERS="${WORKERS:-128}"
        EXECUTOR=(--visible-reasoning)               # gpt-oss thinks in a hidden channel
        BTAG="gptoss20b_high"; BASELINE_ARGS=(--reasoning-effort high) ;;
    qwen)
        MODEL="Qwen/Qwen3.5-35B-A3B-FP8"; BASE_URL="${BASE_URL:-http://localhost:7473/v1}"
        FTAG="qwen35b"; WORKERS="${WORKERS:-64}"
        EXECUTOR=()
        BTAG="qwen35_35b_think"; BASELINE_ARGS=(--family qwen --temperature 1.0 --top-p 0.95) ;;
    qwen9b)                                          # Model_hosting/deploy_qwen35_9b.sh
        MODEL="Qwen/Qwen3.5-9B"; BASE_URL="${BASE_URL:-http://localhost:7472/v1}"
        FTAG="qwen9b"; WORKERS="${WORKERS:-128}"
        EXECUTOR=()
        BTAG="qwen35_9b_think"; BASELINE_ARGS=(--family qwen --temperature 1.0 --top-p 0.95) ;;
    *) echo "family must be gptoss, qwen or qwen9b" >&2; exit 1 ;;
esac
TRAIN="${TRAIN:-datasets/supergpqa_600_train.json}"
TEST="${TEST:-datasets/supergpqa_600_test.json}"
OUT="${OUT:-outputs/pipeline_global_$FTAG}"
RUN="$OUT/run1"
N_DEV="${N_DEV:-100}"
SPLIT_SEED="${SPLIT_SEED:-0}"
GENERATIONS="${GENERATIONS:-10}"
WINDOW="${WINDOW:-32768}"                            # the window the server must report
K=3                                                  # runs per question: baselines and test replicates
PROTOCOLS="direct_high,self_refine_high"            # in-executor rows of the test table
CLUSTER_OUT="outputs/pipeline_cluster_$FTAG"                   # the cluster pipeline's round caches the global ones start from
EXECUTOR+=(--high-cost 3 --turn-cap 15 --any-round-width --last-round-vote)
PYTHON="${PYTHON:-/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python}"
TAGNAME="$("$PYTHON" -c "import sys; sys.path.insert(0, 'scripts'); import program_space as P; print(P.model_tag('$MODEL'))")"
CACHE="$OUT/rounds_$TAGNAME.jsonl"
TEST_CACHE="$RUN/test_eval/rounds_$TAGNAME.jsonl"
SPLITS="$RUN/splits.json"
SEEDS="$RUN/seeds.json"
BRES="baselines/results"
BNAME_TEST="${BTAG}_$(basename "$TEST" .json)"                                # ..._supergpqa_600_test

mkdir -p "$OUT" "$RUN" "$RUN/test_eval" "$BRES"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(stamp)] checking that $BASE_URL serves $MODEL with a $WINDOW-token window"
SERVED="$(curl -sf --max-time 10 "$BASE_URL/models" || true)"
LEN="$(echo "$SERVED" | "$PYTHON" -c "
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
print(next((m.get('max_model_len', '') for m in d.get('data', []) if m.get('id') == '$MODEL'), ''))")"
if [[ -z "$LEN" ]]; then
    echo "no server at $BASE_URL serving $MODEL (is this the right machine and port?)" >&2; exit 1
fi
if [[ "$LEN" != "$WINDOW" ]]; then
    echo "the server's window is $LEN tokens, not $WINDOW: restart it with --max-model-len $WINDOW" \
         "(or set WINDOW=$LEN to run with it; the window is part of every recording's key)" >&2; exit 1
fi
for f in "$TRAIN" "$TEST"; do
    [[ -f "$f" ]] || { echo "$f is missing" >&2; exit 1; }
done
if [[ ! -f "$SEEDS" ]] && ! grep -q "^OPENAI_API_KEY=." .env 2>/dev/null && [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set (nor in .env): the seed stage needs the guide model" >&2; exit 1
fi

# --- the round caches start from copies of the cluster pipeline's (once) -------------------------------------
# (copied to a temporary name first: a copy cut short never passes for a whole one)
if [[ ! -f "$CACHE" && -f "$CLUSTER_OUT/rounds_$TAGNAME.jsonl" ]]; then
    echo "[$(stamp)] $CACHE starts from a copy of $CLUSTER_OUT/rounds_$TAGNAME.jsonl"
    cp "$CLUSTER_OUT/rounds_$TAGNAME.jsonl" "$CACHE.copying" && mv "$CACHE.copying" "$CACHE"
fi
if [[ ! -f "$TEST_CACHE" && -f "$CLUSTER_OUT/run3/test_eval/rounds_$TAGNAME.jsonl" ]]; then
    echo "[$(stamp)] $TEST_CACHE starts from a copy of $CLUSTER_OUT/run3/test_eval/rounds_$TAGNAME.jsonl"
    cp "$CLUSTER_OUT/run3/test_eval/rounds_$TAGNAME.jsonl" "$TEST_CACHE.copying" && mv "$TEST_CACHE.copying" "$TEST_CACHE"
fi

# --- 1. external baselines on the test split -----------------------------------------------------
missing_samples() {  # dataset results -> how many of the K samples per question of the dataset are not done
    "$PYTHON" - "$1" "$2" "$K" <<'EOF'
import json, sys
data, path, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
ids = {r["id"] for r in json.load(open(data))}
done = set()
try:
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("error") is None and r.get("id") in ids and r.get("sample_idx", k) < k:
            done.add((r["id"], r["sample_idx"]))
except FileNotFoundError:
    pass
print(len(ids) * k - len(done))
EOF
}
run_baselines() {  # dataset name -> direct and Self-Refine, K runs each, on exactly the dataset's questions
    local data="$1" name="$2" left_d left_r left_s attempt B k
    local common=(--data "$(readlink -f "$data")" --endpoints "${BASE_URL%/v1}" --model "$MODEL" --k "$K"
                  "${BASELINE_ARGS[@]}")
    for attempt in 1 2 3; do
        left_d="$(missing_samples "$data" "$BRES/direct_$name.jsonl")"
        left_r="$(missing_samples "$data" "$BRES/direct_${name}_rec.jsonl")"
        left_s="$(missing_samples "$data" "$BRES/selfrefine_${name}_rec.jsonl")"
        if [[ "$left_d" == 0 && "$left_r" == 0 && "$left_s" == 0 ]]; then break; fi
        echo "[$(stamp)]   attempt $attempt on $data; to run: direct $left_d, direct with recovery $left_r," \
             "Self-Refine with recovery $left_s samples"
        "$PYTHON" baselines/generate.py --out "$BRES/direct_$name.jsonl" "${common[@]}" --max-tokens 28672
        "$PYTHON" baselines/recover.py --data "$(readlink -f "$data")" --results "$BRES/direct_$name.jsonl" \
            --out "$BRES/direct_${name}_rec.jsonl" --endpoints "${BASE_URL%/v1}" --model "$MODEL" \
            "${BASELINE_ARGS[@]}"
        "$PYTHON" baselines/selfrefine.py --out "$BRES/selfrefine_${name}_rec.jsonl" "${common[@]}" \
            --max-tokens 24576 --feedback-max-tokens 16384 --recover
    done
    left_d="$(missing_samples "$data" "$BRES/direct_$name.jsonl")"
    left_r="$(missing_samples "$data" "$BRES/direct_${name}_rec.jsonl")"
    left_s="$(missing_samples "$data" "$BRES/selfrefine_${name}_rec.jsonl")"
    if [[ "$left_d" != 0 || "$left_r" != 0 || "$left_s" != 0 ]]; then
        echo "external baselines on $data still incomplete (direct $left_d, direct with recovery $left_r," \
             "Self-Refine with recovery $left_s samples): run again" >&2
        exit 1
    fi
    for B in "direct_${name}_rec" "selfrefine_${name}_rec" "direct_$name" "selfrefine_$name"; do
        [[ -f "$BRES/$B.jsonl" && "$(missing_samples "$data" "$BRES/$B.jsonl")" == 0 ]] || continue
        for k in "$K" 1; do
            "$PYTHON" baselines/score.py --data "$data" --results "$BRES/$B.jsonl" --k "$k" \
                --save "$BRES/${B}_k$k.json"
        done
    done
}
external() {  # name -> the label=file pairs the reports read: with recovery, then as published if scored
    local pairs="self-refine=$BRES/selfrefine_$1_rec_k$K.json,direct=$BRES/direct_$1_rec_k$K.json"
    [[ -f "$BRES/selfrefine_$1_k$K.json" ]] && pairs+=",self-refine no recovery=$BRES/selfrefine_$1_k$K.json"
    [[ -f "$BRES/direct_$1_k$K.json" ]] && pairs+=",direct no recovery=$BRES/direct_$1_k$K.json"
    echo "$pairs"
}
echo "[$(stamp)] step 1: external baselines on $TEST"
run_baselines "$TEST" "$BNAME_TEST"

# --- 2. the splits ---------------------------------------------------------------------------------
echo "[$(stamp)] step 2: train / dev split of $TRAIN ($N_DEV dev questions, seed $SPLIT_SEED)"
"$PYTHON" scripts/split_train_dev.py --dataset "$TRAIN" --n-dev "$N_DEV" --seed "$SPLIT_SEED" --out "$SPLITS"

COMMON=(--model "$MODEL" --base-urls "$BASE_URL" --splits "$SPLITS" --dataset "$TRAIN" "${EXECUTOR[@]}")

# --- 3. seeds ----------------------------------------------------------------------------------------
if [[ -f "$SEEDS" ]]; then
    echo "[$(stamp)] step 3: $SEEDS exists, skipping"
else
    echo "[$(stamp)] step 3: seed stage"
    "$PYTHON" scripts/program_seeds_global.py --out "$SEEDS" "${COMMON[@]}" 2>&1 | tee -a "$RUN/seeds.log"
    [[ -f "$SEEDS" ]] || { echo "the seed stage wrote no $SEEDS (see $RUN/seeds.log)" >&2; exit 1; }
fi

# --- 4. search, generation 0 (a finished generation resumes without spending) -------------------------
search() {  # generations -> run or resume the search up to that many generations
    local resume=""
    [[ -f "$RUN/archive.jsonl" ]] && resume="--resume"
    "$PYTHON" scripts/evolve_pipeline_global.py --seeds "$SEEDS" --out "$RUN" --live-cache "$CACHE" \
        --workers "$WORKERS" "${COMMON[@]}" --generations "$1" $resume 2>&1 | tee -a "$RUN/search.log"
}
last_generation() {  # the last generation the search finished (-1 if none; a torn line is skipped)
    "$PYTHON" - "$RUN/generations.jsonl" <<'EOF'
import json, os, sys
gens = []
if os.path.exists(sys.argv[1]):
    for line in open(sys.argv[1]):
        try:
            gens.append(int(json.loads(line)["gen"]))
        except (ValueError, KeyError, TypeError):
            pass
print(max(gens, default=-1))
EOF
}
check_generation() {  # n -> stop unless the search has finished generation n
    local got
    got="$(last_generation)"
    if (( got < $1 )); then
        echo "the search stopped at generation $got, not $1 (see $RUN/search.log); run the script again to resume" >&2
        exit 1
    fi
}
check_seeds() {  # every seed scored on every train question, and every one of those debates ran a round
    "$PYTHON" - "$SEEDS" "$RUN/archive.jsonl" <<'EOF'
import json, sys
seeds = json.load(open(sys.argv[1]))["seeds"]
lines = []
for l in open(sys.argv[2]):
    try:
        lines.append(json.loads(l))
    except ValueError:
        pass                                         # a line torn by a hard stop
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
    got = d["reps"].get("0", {})
    if set(got) != qids:
        bad.append(f"{s['name']}: covers {len(set(got) & qids)} of {len(qids)} train questions")
    elif (dead := sum(v[1] < 1 for v in got.values())):
        bad.append(f"{s['name']}: ran no round on {dead} of {len(qids)} train questions")
if bad:
    print("seed check FAILED (the search is stopped before generation 1):\n  " + "\n  ".join(bad))
    sys.exit(1)
src = {}
for s in seeds:
    src[s["source"]] = src.get(s["source"], 0) + 1
print(f"seed check: all {len(seeds)} seeds {src} ran a round on all {len(qids)} train questions")
EOF
}
echo "[$(stamp)] step 4: search, generation 0 (the seeds on every train question)"
search 0
check_generation 0
check_seeds

# --- 5. search, generations 1..GENERATIONS --------------------------------------------------------
echo "[$(stamp)] step 5: search, generations 1..$GENERATIONS"
search "$GENERATIONS"
check_generation "$GENERATIONS"

# --- 6. the final choice on the dev split ---------------------------------------------------------
# Runs again on every call (free once recorded: the dev debates are cached), so final.json always
# follows the archive as it stands.
echo "[$(stamp)] step 6: the final programs, chosen on the dev split -> $RUN/final.json"
"$PYTHON" scripts/evolve_pipeline_global.py --out "$RUN" --select-dev --live-cache "$CACHE" --workers "$WORKERS" \
    "${COMMON[@]}" 2>&1 | tee -a "$RUN/search.log"
[[ -f "$RUN/final.json" ]] || { echo "no $RUN/final.json (see $RUN/search.log)" >&2; exit 1; }

# --- 7. evaluation on the test split ----------------------------------------------------------------
echo "[$(stamp)] step 7: the final programs and baselines on $TEST, $K replicates"
"$PYTHON" scripts/eval_pipeline_global.py --run "$RUN" --dataset "$TEST" --out "$RUN/test_eval" --live-cache "$TEST_CACHE" \
    --model "$MODEL" --base-urls "$BASE_URL" --workers "$WORKERS" --reps "$K" --baselines "$PROTOCOLS" \
    "${EXECUTOR[@]}" --external "$(external "$BNAME_TEST")" 2>&1 | tee -a "$RUN/test_eval.log"

echo "[$(stamp)] done: $RUN/seeds.md, $RUN/final.md, $RUN/test_eval/results_k$K.md, $RUN/test_eval/accuracy_vs_tokens.svg"

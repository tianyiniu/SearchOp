#!/usr/bin/env bash
# The v3 per-group program search on Qwen/Qwen3.5-35B-A3B-FP8, then the dev evaluation.
# Same search as run_cluster_search_gptoss.sh (see scripts/evolve_program_clusters_v3.py for
# the full account). It has its own output directory, round cache, lock file and archive.
# Resumable: run it again after an interruption.
#
# The champion step (finalists on held-out training questions) is NOT run: once the search has
# its 20 generations the script goes straight to the 300 dev questions with the strongest grid
# program of each group (slot A), the best overall program, and the baselines. The search in
# run1 is already finished, so running this now starts at the dev evaluation.
#
#     mkdir -p outputs/cluster_search_qwen_v3
#     bash run_cluster_search_qwen_v3.sh 2>&1 | tee -a outputs/cluster_search_qwen_v3/pipeline.log
#
# Run it ON THE MACHINE WHERE THE 35B IS SERVED ON PORT 7472 (the alternative environment;
# Model_hosting/deploy_qwen35_35b_fp8.sh). On the machine that serves gpt-oss on 7472 the
# model check below fails and nothing is run. (The 35B on port 7473 of that machine is the
# one the deep-think search uses; BASE_URL=http://localhost:7473/v1 would point this script
# at it instead.)
#
# Seeds: the SAME 24 seed programs as the gpt-oss v3 run are copied in, so the
# two models start from identical programs and can be compared. (A program is
# model-independent; only the sanity run that filtered the random ones was done
# on gpt-oss.) Set SAME_SEEDS=0 to run a fresh seed stage on Qwen instead.
#
# The recordings of the v2 Qwen run are copied in once as a starting cache.
# They only save calls: the search behaves the same with or without them.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
OUT="${OUT:-outputs/cluster_search_qwen_v3}"
RUN="${RUN:-$OUT/run1}"
CACHE="$OUT/rounds_qwen35_35b_a3b_fp8.jsonl"
OLD_CACHE="outputs/cluster_search/rounds_qwen35_35b_a3b_fp8.jsonl"
OLD_LOCK="outputs/cluster_search/rounds_qwen35_35b_a3b_fp8.lock"
GPTOSS_SEEDS="outputs/cluster_search_gptoss_v3/seeds.json"
CLUSTERS="outputs/clusters_train_both_v3.json"
BASE_URL="${BASE_URL:-http://localhost:7472/v1}"   # the 35B in the alternative environment
WORKERS="${WORKERS:-64}"
GENERATIONS="${GENERATIONS:-20}"
TEMPERATURE="${TEMPERATURE:-0.7}"             # as in the v2 Qwen run
SAME_SEEDS="${SAME_SEEDS:-1}"
PYTHON="${PYTHON:-/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python}"
EXTRA="${EXTRA:-}"
REPS=3
DATA="datasets/supergpqa_program_search_dev_small.json"
TEMPLATES="outputs/question_templates_dev_small.jsonl"
VECTORS="outputs/question_vectors_dev_small.npz"
ROUTES="outputs/routes_dev_small.json"
# the independently run baselines (baselines/run_baselines_qwen35b.sh): the ones to report.
# They are added to the dev table if their score files exist.
EXT_SR="baselines/results/selfrefine_qwen35_35b_think_k3.json"
EXT_DIRECT="baselines/results/direct_qwen35_35b_think_k3.json"

case "$OUT" in
    outputs/cluster_search|outputs/cluster_search/*|outputs/cluster_search_gptoss*)
        echo "OUT=$OUT belongs to another run; refusing to write there" >&2; exit 1;;
esac
mkdir -p "$OUT" "$RUN"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(stamp)] checking that $BASE_URL serves $MODEL"
if ! curl -sf --max-time 10 "$BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "no server at $BASE_URL serving $MODEL (is this the right machine and port?)" >&2
    exit 1
fi

COMMON=(--model "$MODEL" --base-urls "$BASE_URL" --live-cache "$CACHE" --clusters "$CLUSTERS"
        --temperature "$TEMPERATURE" --workers "$WORKERS")

# --- 0. search questions ----------------------------------------------------------
if [[ -f "$CLUSTERS" ]]; then
    echo "[$(stamp)] step 0: $CLUSTERS exists, skipping"
else
    echo "[$(stamp)] step 0: drawing the search questions"
    "$PYTHON" scripts/sample_cluster_subsets.py --out "$CLUSTERS"
fi

if [[ ! -f "$CACHE" && -f "$OLD_CACHE" ]]; then
    if [[ -f "$OLD_LOCK" ]]; then
        echo "[$(stamp)] the v2 Qwen run still holds its cache ($OLD_LOCK); starting with an empty cache"
    else
        echo "[$(stamp)] copying the v2 run's recordings to $CACHE as a starting cache"
        cp "$OLD_CACHE" "$CACHE.part" && mv "$CACHE.part" "$CACHE"    # never leave a half copy behind
    fi
fi

# --- 1. seeds -------------------------------------------------------------------
if [[ -f "$OUT/seeds.json" ]]; then
    echo "[$(stamp)] step 1: $OUT/seeds.json exists, skipping"
elif [[ "$SAME_SEEDS" == "1" && -f "$GPTOSS_SEEDS" ]]; then
    echo "[$(stamp)] step 1: copying the gpt-oss v3 run's 24 seed programs"
    cp "$GPTOSS_SEEDS" "$OUT/seeds.json"
else
    echo "[$(stamp)] step 1: fresh seed stage"
    "$PYTHON" scripts/program_seeds_v3.py --out "$OUT/seeds.json" "${COMMON[@]}" 2>&1 | tee -a "$OUT/seeds.log"
fi

# --- 2. search ------------------------------------------------------------------
LAST_GEN=$("$PYTHON" -c "
import json, sys
try:
    lines = [l for l in open('$RUN/generations.jsonl') if l.strip()]
    print(json.loads(lines[-1])['gen'])
except Exception:
    print(-1)")
if [[ "$LAST_GEN" -ge "$GENERATIONS" && -f "$RUN/summary.json" ]]; then
    echo "[$(stamp)] step 2: the search in $RUN has its $GENERATIONS generations, skipping"
else
    if [[ -f "$RUN/archive.jsonl" ]]; then
        echo "[$(stamp)] step 2: resuming the search in $RUN (at generation $LAST_GEN)"
        RESUME="--resume"
    else
        echo "[$(stamp)] step 2: starting the search in $RUN"
        RESUME=""
    fi
    "$PYTHON" scripts/evolve_program_clusters_v3.py --seeds "$OUT/seeds.json" --out "$RUN" \
        "${COMMON[@]}" --generations "$GENERATIONS" $RESUME $EXTRA 2>&1 | tee -a "$RUN/search.log"
fi

# --- 3. dev evaluation (no champion step) -------------------------------------------
# The dev descriptions, vectors and routes do not depend on the debate model; the ones an
# earlier dev evaluation made are reused.
echo "[$(stamp)] step 3a: describing the dev questions (skips the ones already described)"
"$PYTHON" scripts/describe_questions.py --anchors outputs/anchors.json --dataset "$DATA" \
    --out "$TEMPLATES" --workers 8
if [[ -f "$VECTORS" && "$VECTORS" -nt "$TEMPLATES" ]]; then
    echo "[$(stamp)] step 3b: $VECTORS is up to date, skipping"
else
    echo "[$(stamp)] step 3b: embedding the dev questions (CPU)"
    "$PYTHON" scripts/embed_questions.py --templates "$TEMPLATES" --dataset "$DATA" --out "$VECTORS" --device cpu
fi
echo "[$(stamp)] step 3c: routing (nearest medoid, no model)"
"$PYTHON" scripts/route_questions.py --clusters "$CLUSTERS" --test-vectors "$VECTORS" --out "$ROUTES"

EXTERNAL=""
[[ -f "$EXT_SR" ]] && EXTERNAL="self-refine (thinking on)=$EXT_SR"
[[ -f "$EXT_DIRECT" ]] && EXTERNAL="${EXTERNAL:+$EXTERNAL,}direct (thinking on)=$EXT_DIRECT"
[[ -n "$EXTERNAL" ]] || echo "[$(stamp)] note: no independently run baselines found yet (baselines/run_baselines_qwen35b.sh); the table will have in-executor baselines only"

# Replicate 1 of everything first: results_k1.md is written after about a third of the work.
echo "[$(stamp)] step 3d: grid programs and baselines on the dev questions, $REPS replicates, no champions"
"$PYTHON" scripts/eval_routed_dev.py --run "$RUN" --routes "$ROUTES" --dataset "$DATA" --no-champions \
    --model "$MODEL" --base-urls "$BASE_URL" --temperature "$TEMPERATURE" --workers "$WORKERS" \
    --reps "$REPS" --external "$EXTERNAL" 2>&1 | tee -a "$RUN/dev_eval.log"

echo "[$(stamp)] done: $RUN/dev_eval/results_k1.md ... results_k$REPS.md"

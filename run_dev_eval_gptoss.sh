#!/usr/bin/env bash
# Test-time results for the v3 gpt-oss-20b search on the 300 dev questions.
# Four steps: describe, embed, route, evaluate. Resumable: run it again after
# an interruption; finished steps are skipped and recorded debates are reused.
#
#     bash run_dev_eval_gptoss.sh 2>&1 | tee -a outputs/cluster_search_gptoss_v3/run1/dev_eval.log
#
# Run it ON THE SERVER THAT HOSTS gpt-oss (the model is bound to localhost).
# Step 1 calls the labelling model (one short call per dev question; the key is
# read from .env). Step 2 runs the 0.6B embedding model on the CPU, so it takes
# no GPU memory from the debate server. Nothing in the search run is written to
# except the new dev_eval/ directory inside it.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="openai/gpt-oss-20b"
RUN="outputs/cluster_search_gptoss_v3/run1"
DATA="datasets/supergpqa_program_search_dev_small.json"
CLUSTERS="outputs/clusters_train_both_v3.json"
TEMPLATES="outputs/question_templates_dev_small.jsonl"
VECTORS="outputs/question_vectors_dev_small.npz"
ROUTES="outputs/routes_dev_small.json"
BASE_URL="http://localhost:7472/v1"
WORKERS=128
TEMPERATURE=1.0                                # as in the search
REPS=3                                         # run one after another; tables after each
PYTHON="/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python"
# the independently run baselines (baselines/run_baselines_gptoss.sh): the ones to report
EXTERNAL="self-refine (reasoning high)=baselines/results/selfrefine_gptoss20b_high_k3.json,direct (reasoning high)=baselines/results/direct_gptoss20b_high_k3.json"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

[[ -f "$RUN/champions.json" ]] || { echo "$RUN/champions.json is missing: the search has not finished" >&2; exit 1; }
echo "[$(stamp)] checking that $BASE_URL serves $MODEL"
if ! curl -sf --max-time 10 "$BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "no server at $BASE_URL serving $MODEL (is this the right machine and port?)" >&2
    exit 1
fi

# --- 1. describe the dev questions (same prompt, anchors and model as the train split) ---
# Skips the questions already described, so it is safe to run again.
echo "[$(stamp)] step 1: describing the dev questions"
"$PYTHON" scripts/describe_questions.py --anchors outputs/anchors.json --dataset "$DATA" \
    --out "$TEMPLATES" --workers 8

# --- 2. embed them ---------------------------------------------------------------
if [[ -f "$VECTORS" && "$VECTORS" -nt "$TEMPLATES" ]]; then      # redone if step 1 added descriptions
    echo "[$(stamp)] step 2: $VECTORS is up to date, skipping"
else
    echo "[$(stamp)] step 2: embedding the dev questions (CPU)"
    "$PYTHON" scripts/embed_questions.py --templates "$TEMPLATES" --dataset "$DATA" --out "$VECTORS" --device cpu
fi

# --- 3. route: nearest medoid (no model) -------------------------------------------
echo "[$(stamp)] step 3: routing"
"$PYTHON" scripts/route_questions.py --clusters "$CLUSTERS" --test-vectors "$VECTORS" --out "$ROUTES"

# --- 4. run the champions, grid programs and baselines on every dev question --------
# Replicate 1 of everything first: results_k1.md (the pass@1 numbers) is written
# after about a third of the work, then results_k2.md, then results_k3.md.
echo "[$(stamp)] step 4: champions and baselines on the dev questions"
"$PYTHON" scripts/eval_routed_dev.py --run "$RUN" --routes "$ROUTES" --dataset "$DATA" \
    --model "$MODEL" --base-urls "$BASE_URL" --temperature "$TEMPERATURE" --workers "$WORKERS" \
    --reps "$REPS" --external "$EXTERNAL" --visible-reasoning

echo "[$(stamp)] done: $RUN/dev_eval/results_k$REPS.md (first replicate alone: results_k1.md)"

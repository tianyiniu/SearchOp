#!/usr/bin/env bash
# Dev evaluation of the gpt-oss-20b deep-think search, WITHOUT the champion step.
# The search (outputs/cluster_search_gptoss_deep/run1) finished; champion picking was cancelled
# and is skipped here. What is run on the 300 dev questions, 3 replicates one after another:
#   - the strongest grid program of each group (the six slot-A holders), routed by nearest medoid
#   - the program with the best score over all search questions, used for every question
#   - the baselines direct, mad, self_refine and deep_direct, under the same executor settings
# Resumable: recorded debates are reused, so run it again after an interruption.
#
#     bash run_dev_eval_gptoss_deep.sh 2>&1 | tee -a outputs/cluster_search_gptoss_deep/run1/dev_eval.log
#
# Run it ON THE SERVER THAT HOSTS gpt-oss (the model is bound to localhost).
set -euo pipefail
cd "$(dirname "$0")"

MODEL="openai/gpt-oss-20b"
RUN="outputs/cluster_search_gptoss_deep/run1"
CLUSTERS="outputs/clusters_train_both_v3.json"
BASE_URL="http://localhost:7472/v1"
WORKERS=128
TEMPERATURE=1.0
REPS=3
BASELINES="direct,mad,self_refine,deep_direct"
DATA="datasets/supergpqa_program_search_dev_small.json"
TEMPLATES="outputs/question_templates_dev_small.jsonl"
VECTORS="outputs/question_vectors_dev_small.npz"
ROUTES="outputs/routes_dev_small.json"
PYTHON="/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python"
# the executor settings of the search; the evaluation refuses to run if they differ
EXECUTOR=(--visible-reasoning --deep-think --summary-words 500 --digest-head 2000 --digest-tail 2000)
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

[[ -f "$RUN/summary.json" ]] || { echo "$RUN/summary.json is missing: the search has not finished" >&2; exit 1; }
echo "[$(stamp)] checking that $BASE_URL serves $MODEL"
if ! curl -sf --max-time 10 "$BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "no server at $BASE_URL serving $MODEL (is this the right machine and port?)" >&2
    exit 1
fi

# The dev descriptions, vectors and routes were made by the first gpt-oss dev evaluation and do
# not depend on the debate model; each step below only does work if something is missing.
echo "[$(stamp)] describing the dev questions (skips the ones already described)"
"$PYTHON" scripts/describe_questions.py --anchors outputs/anchors.json --dataset "$DATA" \
    --out "$TEMPLATES" --workers 8
if [[ -f "$VECTORS" && "$VECTORS" -nt "$TEMPLATES" ]]; then
    echo "[$(stamp)] $VECTORS is up to date, skipping"
else
    echo "[$(stamp)] embedding the dev questions (CPU)"
    "$PYTHON" scripts/embed_questions.py --templates "$TEMPLATES" --dataset "$DATA" --out "$VECTORS" --device cpu
fi
echo "[$(stamp)] routing (nearest medoid, no model)"
"$PYTHON" scripts/route_questions.py --clusters "$CLUSTERS" --test-vectors "$VECTORS" --out "$ROUTES"

echo "[$(stamp)] grid programs and baselines on the dev questions, $REPS replicates, no champions"
"$PYTHON" scripts/eval_routed_dev.py --run "$RUN" --routes "$ROUTES" --dataset "$DATA" --no-champions \
    --model "$MODEL" --base-urls "$BASE_URL" --temperature "$TEMPERATURE" --workers "$WORKERS" \
    --reps "$REPS" --baselines "$BASELINES" "${EXECUTOR[@]}"

echo "[$(stamp)] done: $RUN/dev_eval/results_k1.md ... results_k$REPS.md"

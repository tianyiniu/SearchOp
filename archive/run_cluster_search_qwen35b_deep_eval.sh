#!/usr/bin/env bash
# Dev evaluation ONLY for the Qwen 3.5 35B deep-think search, skipping the champion step
# (step 3 of run_cluster_search_qwen35b_deep.sh). The routed programs are the strongest grid
# programs (slot A of every group, as the search left them) and the one-program-for-everyone
# row is the program with the best score over all search questions (summary.json, top_overall),
# exactly as the gpt-oss deep run was evaluated.
#
# Use this instead of letting the original driver run its step 3 (about 5 hours on the 35B):
#   1. on bansal27, stop the running driver and its champion step
#        (the lock file names the pid: cat outputs/cluster_search_qwen35b_deep/rounds_qwen35_35b_a3b_fp8.lock)
#   2. on the same machine (the 35B is bound to localhost:7473):
#        bash run_cluster_search_qwen35b_deep_eval.sh 2>&1 | tee -a outputs/cluster_search_qwen35b_deep/pipeline.log
#
# Same executor settings and the same dev files as the driver. Resumable: the dev debates are
# cached in $RUN/dev_eval/rounds_<model>.jsonl and the tables are rewritten after each replicate.
# The independently run thinking-on baselines (baselines/results/*_qwen35_35b_think_k3.json) are
# shown beside ours when present; those are the numbers to report.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
OUT="outputs/cluster_search_qwen35b_deep"
RUN="$OUT/run1"
CLUSTERS="outputs/clusters_train_both_v3.json"
BASE_URL="http://localhost:7473/v1"
WORKERS=64
TEMPERATURE=0.7
REPS=3
BASELINES="direct,mad,self_refine,deep_direct"
DATA="datasets/supergpqa_program_search_dev_small.json"
TEMPLATES="outputs/question_templates_dev_small.jsonl"
VECTORS="outputs/question_vectors_dev_small.npz"
ROUTES="outputs/routes_dev_small.json"
EXT_SR="baselines/results/selfrefine_qwen35_35b_think_k3.json"
EXT_DIRECT="baselines/results/direct_qwen35_35b_think_k3.json"
PYTHON="/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
echo "[$(stamp)] checking that $BASE_URL serves $MODEL"
if ! curl -sf --max-time 10 "$BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "no server at $BASE_URL serving $MODEL (is this the right machine and port?)" >&2
    exit 1
fi
[[ -f "$RUN/summary.json" ]] || { echo "$RUN/summary.json is missing: the search has not finished" >&2; exit 1; }
if [[ -f "$OUT/rounds_qwen35_35b_a3b_fp8.lock" ]]; then
    echo "note: the search cache lock is still present ($(cat "$OUT/rounds_qwen35_35b_a3b_fp8.lock"));"
    echo "      the dev evaluation uses its own cache under $RUN/dev_eval, so this is fine once that process is stopped"
fi

EXECUTOR=(--executor v2 --deep-think --summary-words 500 --digest-head 2000 --digest-tail 2000)

echo "[$(stamp)] step 4a: describing the dev questions (skips the ones already described)"
"$PYTHON" scripts/describe_questions.py --anchors outputs/anchors.json --dataset "$DATA" \
    --out "$TEMPLATES" --workers 8
if [[ -f "$VECTORS" && "$VECTORS" -nt "$TEMPLATES" ]]; then
    echo "[$(stamp)] step 4b: $VECTORS is up to date, skipping"
else
    echo "[$(stamp)] step 4b: embedding the dev questions (CPU)"
    "$PYTHON" scripts/embed_questions.py --templates "$TEMPLATES" --dataset "$DATA" --out "$VECTORS" --device cpu
fi
echo "[$(stamp)] step 4c: routing (nearest medoid, no model)"
"$PYTHON" scripts/route_questions.py --clusters "$CLUSTERS" --test-vectors "$VECTORS" --out "$ROUTES"

EXTERNAL=""
[[ -f "$EXT_SR" ]] && EXTERNAL="self-refine (thinking on)=$EXT_SR"
[[ -f "$EXT_DIRECT" ]] && EXTERNAL="${EXTERNAL:+$EXTERNAL,}direct (thinking on)=$EXT_DIRECT"
[[ -n "$EXTERNAL" ]] || echo "[$(stamp)] note: no independently run baselines found; the table will have in-executor baselines only"

echo "[$(stamp)] step 4d: strongest grid programs and baselines on the dev questions, $REPS replicates, no champion step"
"$PYTHON" scripts/eval_routed_dev.py --run "$RUN" --routes "$ROUTES" --dataset "$DATA" --no-champions \
    --model "$MODEL" --base-urls "$BASE_URL" --temperature "$TEMPERATURE" --workers "$WORKERS" \
    --reps "$REPS" --baselines "$BASELINES" ${EXTERNAL:+--external "$EXTERNAL"} "${EXECUTOR[@]}" \
    2>&1 | tee -a "$RUN/dev_eval.log"

echo "[$(stamp)] done: $RUN/dev_eval/results_k$REPS.md"

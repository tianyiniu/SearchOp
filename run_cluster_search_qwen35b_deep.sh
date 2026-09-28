#!/usr/bin/env bash
# The v3 per-group program search on Qwen/Qwen3.5-35B-A3B-FP8 WITH the deep-think speaker and
# longer summaries, from the seeds to the dev evaluation. Resumable: run it again after an
# interruption.
#
#     mkdir -p outputs/cluster_search_qwen35b_deep
#     bash run_cluster_search_qwen35b_deep.sh 2>&1 | tee -a outputs/cluster_search_qwen35b_deep/pipeline.log
#
# Run it ON THE SERVER THAT HOSTS the 35B (it is bound to localhost, port 7473).
#
# Same executor settings as run_cluster_search_gptoss_deep.sh, minus --visible-reasoning (that
# flag is for gpt-oss, whose ordinary replies hide their reasoning):
#   --deep-think          the deep_think speaker as a plan round and a move. On Qwen "the highest
#                         reasoning setting" means thinking mode ON (Qwen has no graded effort:
#                         thinking is on or off) and no reply cap, so the server lets it use what
#                         is left of its 32,768-token window. Every other speaker runs with
#                         thinking OFF and a 6,144-token reply, as in all earlier Qwen runs.
#                         Checked live on this server: the deep-think request came back with
#                         19k-38k characters of thinking; ordinary calls came back with none
#                         (except that the model opens a thinking block on its own in roughly 1
#                         ordinary call in 12, which no switch prevents).
#                         One deep-think turn counts as 5 turns. deep_direct (one deep-think turn
#                         and nothing else) takes direct's place among the 8 literature seeds.
#   --summary-words 500   comprehensive summary of up to 500 words; a reply already within 500
#                         words that commits is shown as it is, with no summary call
#   --digest 2000+2000    the window for a reply shown without a summary
# Every recording is new (these settings change what each speaker is shown), so no earlier Qwen
# cache is reused. Each recording carries its token usage per speaker.
#
# Seeds: the SAME 24 seed programs as the gpt-oss deep run are copied in if that run has made
# them, so the two models start from identical programs. Set SAME_SEEDS=0 below for a fresh seed
# stage on Qwen instead.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
OUT="outputs/cluster_search_qwen35b_deep"
RUN="$OUT/run1"
CACHE="$OUT/rounds_qwen35_35b_a3b_fp8.jsonl"
CLUSTERS="outputs/clusters_train_both_v3.json"
GPTOSS_SEEDS="outputs/cluster_search_gptoss_deep/seeds.json"
SAME_SEEDS=1
BASE_URL="http://localhost:7473/v1"
WORKERS=64
GENERATIONS=20
TEMPERATURE=0.7                                # as in every earlier Qwen search
REPS=3
BASELINES="direct,mad,self_refine,deep_direct"
DATA="datasets/supergpqa_program_search_dev_small.json"
TEMPLATES="outputs/question_templates_dev_small.jsonl"
VECTORS="outputs/question_vectors_dev_small.npz"
ROUTES="outputs/routes_dev_small.json"
PYTHON="/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python"

mkdir -p "$OUT" "$RUN"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(stamp)] checking that $BASE_URL serves $MODEL"
if ! curl -sf --max-time 10 "$BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "no server at $BASE_URL serving $MODEL (is this the right machine and port?)" >&2
    exit 1
fi
[[ -f "$CLUSTERS" ]] || { echo "$CLUSTERS is missing; run scripts/sample_cluster_subsets.py first" >&2; exit 1; }

EXECUTOR=(--deep-think --summary-words 500 --digest-head 2000 --digest-tail 2000)
COMMON=(--model "$MODEL" --base-urls "$BASE_URL" --live-cache "$CACHE" --clusters "$CLUSTERS"
        --temperature "$TEMPERATURE" --workers "$WORKERS" "${EXECUTOR[@]}")

# --- 1. seeds -------------------------------------------------------------------
if [[ -f "$OUT/seeds.json" ]]; then
    echo "[$(stamp)] step 1: $OUT/seeds.json exists, skipping"
elif [[ "$SAME_SEEDS" == "1" && -f "$GPTOSS_SEEDS" ]]; then
    echo "[$(stamp)] step 1: copying the gpt-oss deep run's 24 seed programs"
    cp "$GPTOSS_SEEDS" "$OUT/seeds.json"
else
    echo "[$(stamp)] step 1: fresh seed stage (6 model-written + 8 literature + 10 random)"
    "$PYTHON" scripts/program_seeds_v3.py --out "$OUT/seeds.json" "${COMMON[@]}" 2>&1 | tee -a "$OUT/seeds.log"
fi

# --- 2, 3. search and champions ---------------------------------------------------
if [[ -f "$RUN/champions.json" ]]; then
    echo "[$(stamp)] steps 2, 3: $RUN/champions.json exists, the search is finished; skipping"
else
    if [[ -f "$RUN/archive.jsonl" ]]; then
        echo "[$(stamp)] step 2: resuming the search in $RUN"
        RESUME="--resume"
    else
        echo "[$(stamp)] step 2: starting the search in $RUN"
        RESUME=""
    fi
    "$PYTHON" scripts/evolve_program_clusters_v3.py --seeds "$OUT/seeds.json" --out "$RUN" \
        "${COMMON[@]}" --generations "$GENERATIONS" $RESUME 2>&1 | tee -a "$RUN/search.log"

    echo "[$(stamp)] step 3: champions and baselines on held-out questions"
    "$PYTHON" scripts/evolve_program_clusters_v3.py --out "$RUN" --pick-champions "${COMMON[@]}" \
        --baselines "$BASELINES" 2>&1 | tee -a "$RUN/champions.log"
fi

# --- 4. test time -----------------------------------------------------------------
# The dev descriptions, vectors and routes do not depend on the debate model or the executor;
# the ones an earlier dev evaluation made are reused.
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

echo "[$(stamp)] step 4d: routed programs and baselines on the dev questions, $REPS replicates"
"$PYTHON" scripts/eval_routed_dev.py --run "$RUN" --routes "$ROUTES" --dataset "$DATA" \
    --model "$MODEL" --base-urls "$BASE_URL" --temperature "$TEMPERATURE" --workers "$WORKERS" \
    --reps "$REPS" --baselines "$BASELINES" "${EXECUTOR[@]}" 2>&1 | tee -a "$RUN/dev_eval.log"

echo "[$(stamp)] done: $RUN/summary.json, $RUN/champions.json, $RUN/dev_eval/results_k$REPS.md"

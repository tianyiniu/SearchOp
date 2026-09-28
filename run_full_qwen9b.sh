#!/usr/bin/env bash
# Everything for Qwen/Qwen3.5-9B in one go:
#
#   A. baselines on the 300 dev questions: direct chain-of-thought and Self-Refine, k=3
#   B. the v3 per-group program search: seeds, 20 generations, champions
#   C. test time: describe + embed + route the dev questions, then run the routed
#      programs and the debate-format baselines on them, 3 replicates one after another
#
#     mkdir -p outputs/cluster_search_qwen9b_v3
#     bash run_full_qwen9b.sh 2>&1 | tee -a outputs/cluster_search_qwen9b_v3/pipeline.log
#
# Run it ON THE SERVER THAT HOSTS the 9B (Model_hosting/deploy_qwen35_9b.sh binds it to
# localhost:7472). Resumable: run it again after an interruption. Finished baseline samples,
# finished steps and recorded debates are all reused.
#
# Two model settings are in play, on purpose:
#   baselines (part A)  thinking ON, the Qwen model-card sampling (temperature 1.0, top_p 0.95,
#                       top_k 20, presence penalty 1.5), long replies: the collaborator's setup,
#                       the counterpart of "reasoning high" in the gpt-oss baselines
#   search + dev (B, C) thinking OFF (the server default), temperature 0.7, 6144-token replies:
#                       the setup of every program search so far
# Part C also runs direct / mad / self_refine as debate programs under the search's settings,
# so there is a like-for-like comparison beside the thinking-on baselines.
#
# Seeds: the same 24 seed programs as the gpt-oss v3 run are copied in, so the models start
# from identical programs. Set SAME_SEEDS=0 below for a fresh seed stage on the 9B instead.
# The dev descriptions, vectors and routes do not depend on the debate model; if the gpt-oss
# dev evaluation already made them they are reused.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3.5-9B"
ENDPOINT="http://localhost:7472"
BASE_URL="$ENDPOINT/v1"
PYTHON="/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python"

# part A
K=3
TAG="qwen35_9b_think"
RESULTS="baselines/results"

# parts B and C
OUT="outputs/cluster_search_qwen9b_v3"
RUN="$OUT/run1"
CACHE="$OUT/rounds_qwen35_9b.jsonl"
CLUSTERS="outputs/clusters_train_both_v3.json"
GPTOSS_SEEDS="outputs/cluster_search_gptoss_v3/seeds.json"
SAME_SEEDS=1
WORKERS=128
GENERATIONS=20
TEMPERATURE=0.7
REPS=3
DATA="datasets/supergpqa_program_search_dev_small.json"
TEMPLATES="outputs/question_templates_dev_small.jsonl"
VECTORS="outputs/question_vectors_dev_small.npz"
ROUTES="outputs/routes_dev_small.json"

mkdir -p "$OUT" "$RUN" "$RESULTS"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(stamp)] checking that $BASE_URL serves $MODEL"
if ! curl -sf --max-time 10 "$BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "no server at $BASE_URL serving $MODEL (is this the right machine and port?)" >&2
    exit 1
fi

# =============================== A. baselines ===================================
# The server's window is 32,768 tokens, so max_tokens stays below it (the scripts' defaults).
QWEN=(--family qwen --temperature 1.0 --top-p 0.95 --endpoints "$ENDPOINT" --model "$MODEL" --k "$K")

echo "[$(stamp)] A1: direct chain-of-thought, thinking on, k=$K"
"$PYTHON" baselines/generate.py --out "$RESULTS/direct_$TAG.jsonl" "${QWEN[@]}"
echo "[$(stamp)] A2: Self-Refine, thinking on, k=$K"
"$PYTHON" baselines/selfrefine.py --out "$RESULTS/selfrefine_$TAG.jsonl" "${QWEN[@]}"
for NAME in direct selfrefine; do
    echo; echo "##### $NAME: all $K runs"
    "$PYTHON" baselines/score.py --results "$RESULTS/${NAME}_$TAG.jsonl" --k "$K" --save "$RESULTS/${NAME}_${TAG}_k$K.json"
    echo; echo "##### $NAME: first run only"
    "$PYTHON" baselines/score.py --results "$RESULTS/${NAME}_$TAG.jsonl" --k 1 --save "$RESULTS/${NAME}_${TAG}_k1.json"
done

# =============================== B. program search ==============================
COMMON=(--model "$MODEL" --base-urls "$BASE_URL" --live-cache "$CACHE" --clusters "$CLUSTERS"
        --temperature "$TEMPERATURE" --workers "$WORKERS")

if [[ -f "$CLUSTERS" ]]; then
    echo "[$(stamp)] B0: $CLUSTERS exists, skipping"
else
    echo "[$(stamp)] B0: drawing the search questions"
    "$PYTHON" scripts/sample_cluster_subsets.py --out "$CLUSTERS"
fi

if [[ -f "$OUT/seeds.json" ]]; then
    echo "[$(stamp)] B1: $OUT/seeds.json exists, skipping"
elif [[ "$SAME_SEEDS" == "1" && -f "$GPTOSS_SEEDS" ]]; then
    echo "[$(stamp)] B1: copying the gpt-oss v3 run's 24 seed programs"
    cp "$GPTOSS_SEEDS" "$OUT/seeds.json"
else
    echo "[$(stamp)] B1: fresh seed stage"
    "$PYTHON" scripts/program_seeds_v3.py --out "$OUT/seeds.json" "${COMMON[@]}" 2>&1 | tee -a "$OUT/seeds.log"
fi

if [[ -f "$RUN/champions.json" ]]; then
    echo "[$(stamp)] B2, B3: $RUN/champions.json exists, the search is finished; skipping"
else
    if [[ -f "$RUN/archive.jsonl" ]]; then
        echo "[$(stamp)] B2: resuming the search in $RUN"
        RESUME="--resume"
    else
        echo "[$(stamp)] B2: starting the search in $RUN"
        RESUME=""
    fi
    "$PYTHON" scripts/evolve_program_clusters_v3.py --seeds "$OUT/seeds.json" --out "$RUN" \
        "${COMMON[@]}" --generations "$GENERATIONS" $RESUME 2>&1 | tee -a "$RUN/search.log"

    echo "[$(stamp)] B3: champions and literature baselines on held-out questions"
    "$PYTHON" scripts/evolve_program_clusters_v3.py --out "$RUN" --pick-champions "${COMMON[@]}" \
        2>&1 | tee -a "$RUN/champions.log"
fi

# =============================== C. test time ===================================
echo "[$(stamp)] C1: describing the dev questions (skips the ones already described)"
"$PYTHON" scripts/describe_questions.py --anchors outputs/anchors.json --dataset "$DATA" \
    --out "$TEMPLATES" --workers 8

if [[ -f "$VECTORS" && "$VECTORS" -nt "$TEMPLATES" ]]; then      # redone if C1 added descriptions
    echo "[$(stamp)] C2: $VECTORS is up to date, skipping"
else
    echo "[$(stamp)] C2: embedding the dev questions (CPU)"
    "$PYTHON" scripts/embed_questions.py --templates "$TEMPLATES" --dataset "$DATA" --out "$VECTORS" --device cpu
fi

echo "[$(stamp)] C3: routing (nearest medoid, no model)"
"$PYTHON" scripts/route_questions.py --clusters "$CLUSTERS" --test-vectors "$VECTORS" --out "$ROUTES"

# Replicate 1 of everything first: results_k1.md is written after about a third of the work.
echo "[$(stamp)] C4: routed programs and baselines on the dev questions, $REPS replicates"
"$PYTHON" scripts/eval_routed_dev.py --run "$RUN" --routes "$ROUTES" --dataset "$DATA" \
    --model "$MODEL" --base-urls "$BASE_URL" --temperature "$TEMPERATURE" --workers "$WORKERS" \
    --reps "$REPS" 2>&1 | tee -a "$RUN/dev_eval.log"

echo "[$(stamp)] done"
echo "  baselines:  $RESULTS/{direct,selfrefine}_${TAG}_k{1,$K}.json"
echo "  search:     $RUN/summary.json, $RUN/champions.json"
echo "  dev:        $RUN/dev_eval/results_k1.md ... results_k$REPS.md"

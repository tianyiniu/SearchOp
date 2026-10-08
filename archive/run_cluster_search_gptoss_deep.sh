#!/usr/bin/env bash
# The v3 per-group program search on openai/gpt-oss-20b WITH the deep-think speaker and
# longer summaries, from the seed stage to the dev evaluation. Resumable: run it again after
# an interruption.
#
#     mkdir -p outputs/cluster_search_gptoss_deep
#     bash run_cluster_search_gptoss_deep.sh 2>&1 | tee -a outputs/cluster_search_gptoss_deep/pipeline.log
#
# Run it ON THE SERVER THAT HOSTS gpt-oss (the model is bound to localhost).
#
# What differs from run_cluster_search_gptoss.sh (everything else is the same v3 search):
#   --deep-think          one more speaker, deep_think, as a plan round and as a move: the model's
#                         highest reasoning setting and no reply cap (the server lets it use what
#                         is left of its 32,768-token window). One deep-think turn counts as 5
#                         turns. deep_direct (one deep-think turn, nothing else) takes the place
#                         of direct (one solver) among the 8 literature seeds, so there are still
#                         24 seeds; both are scored as baselines.
#   --summary-words 500   the summary later speakers read may run to 500 words (was 120) and is
#                         asked to be comprehensive: steps in order, calculations, assumptions
#                         A reply already within 500 words that commits is shown as it is,
#                         with no summary call (deep_think is always summarised).
#   --digest 2000+2000    the window applied to a reply shown WITHOUT a summary (a summary is
#                         always shown whole). Widened from 300+900 characters so a 500-word
#                         reply, ~3,300 characters, is not cut in the middle.
# These change what every speaker is shown, so every recording is new: no earlier cache can be
# reused, and the seed stage is run afresh (the model-written seeds are told about deep_think).
# The search questions, the groups and the dev routes are the same as the other v3 runs.
#
# Expect this to be several times slower than the plain gpt-oss search: a deep-think call runs
# to ~10,000 tokens where an ordinary reply is ~2,000.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="openai/gpt-oss-20b"
OUT="outputs/cluster_search_gptoss_deep"
RUN="$OUT/run1"
CACHE="$OUT/rounds_gpt_oss_20b.jsonl"
CLUSTERS="outputs/clusters_train_both_v3.json"
BASE_URL="http://localhost:7472/v1"
WORKERS=128
GENERATIONS=20
TEMPERATURE=1.0                                # OpenAI's recommended sampling for gpt-oss
REPS=3
BASELINES="direct,mad,self_refine,deep_direct"
DATA="datasets/supergpqa_program_search_dev_small.json"
TEMPLATES="outputs/question_templates_dev_small.jsonl"
VECTORS="outputs/question_vectors_dev_small.npz"
ROUTES="outputs/routes_dev_small.json"
PYTHON="/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python"
# the independently run baselines (baselines/run_baselines_gptoss.sh): the ones to report
EXTERNAL="self-refine (reasoning high)=baselines/results/selfrefine_gptoss20b_high_k3.json,direct (reasoning high)=baselines/results/direct_gptoss20b_high_k3.json"

mkdir -p "$OUT" "$RUN"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(stamp)] checking that $BASE_URL serves $MODEL"
if ! curl -sf --max-time 10 "$BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "no server at $BASE_URL serving $MODEL (is this the right machine and port?)" >&2
    exit 1
fi
[[ -f "$CLUSTERS" ]] || { echo "$CLUSTERS is missing; run scripts/sample_cluster_subsets.py first" >&2; exit 1; }

EXECUTOR=(--executor v2 --visible-reasoning --deep-think --summary-words 500 --digest-head 2000 --digest-tail 2000)
COMMON=(--model "$MODEL" --base-urls "$BASE_URL" --live-cache "$CACHE" --clusters "$CLUSTERS"
        --temperature "$TEMPERATURE" --workers "$WORKERS" "${EXECUTOR[@]}")

# --- 1. seeds -------------------------------------------------------------------
if [[ -f "$OUT/seeds.json" ]]; then
    echo "[$(stamp)] step 1: $OUT/seeds.json exists, skipping"
else
    echo "[$(stamp)] step 1: seed stage (6 model-written + 8 literature + 10 random)"
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
# the ones the first gpt-oss dev evaluation made are reused.
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
    --reps "$REPS" --baselines "$BASELINES" --external "$EXTERNAL" "${EXECUTOR[@]}" 2>&1 | tee -a "$RUN/dev_eval.log"

echo "[$(stamp)] done: $RUN/summary.json, $RUN/champions.json, $RUN/dev_eval/results_k$REPS.md"

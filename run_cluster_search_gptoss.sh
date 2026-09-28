#!/usr/bin/env bash
# The v3 per-group program search on openai/gpt-oss-20b.
# Four steps: search questions, seeds, search, champions. It has its own output
# directory, round cache, lock file and archive, and shares nothing on disk
# with the Qwen run or with the earlier (v2) gpt-oss run except files it only
# reads. Resumable: run it again after an interruption.
#
#     bash run_cluster_search_gptoss.sh 2>&1 | tee -a outputs/cluster_search_gptoss_v3/pipeline.log
#
# Run it ON THE SERVER THAT HOSTS gpt-oss: deploy_gpt_oss_20b.sh binds the
# model to localhost, so it is not reachable from another machine.
#
# What v3 is (scripts/evolve_program_clusters_v3.py has the full account):
#   questions  50 per group, a random draw weighted towards the group's centre
#   seeds      4k = one model-written per group + 8 literature + random (24 at k=6)
#   parents    2 per group: the strongest, and the rank-gap specialist (12)
#   children   2 per parent, one uniformly drawn random edit each (24)
#   screen     a child leaves its target group only if within one paired
#              standard error of its parent there
#   stop       a fixed 20 generations
#
# The recordings of the v2 gpt-oss run are copied in once as a starting cache.
# They only save calls: a recording is keyed by question, round sequence and
# replicate, so the search behaves the same with or without them.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="openai/gpt-oss-20b"
OUT="${OUT:-outputs/cluster_search_gptoss_v3}"
RUN="${RUN:-$OUT/run1}"
CACHE="$OUT/rounds_gpt_oss_20b.jsonl"
OLD_CACHE="outputs/cluster_search_gptoss/rounds_gpt_oss_20b.jsonl"
CLUSTERS="outputs/clusters_train_both_v3.json"
BASE_URL="${BASE_URL:-http://localhost:7472/v1}"
WORKERS="${WORKERS:-128}"
GENERATIONS="${GENERATIONS:-20}"
TEMPERATURE="${TEMPERATURE:-1.0}"             # OpenAI's recommended sampling for gpt-oss
PYTHON="${PYTHON:-/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python}"
EXTRA="${EXTRA:-}"

case "$OUT" in
    outputs/cluster_search|outputs/cluster_search/*|outputs/cluster_search_gptoss|outputs/cluster_search_gptoss/*)
        echo "OUT=$OUT belongs to an earlier run; refusing to write there" >&2; exit 1;;
esac
mkdir -p "$OUT" "$RUN"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(stamp)] checking that $BASE_URL serves $MODEL"
if ! curl -sf --max-time 10 "$BASE_URL/models" | grep -q "\"$MODEL\""; then
    echo "no server at $BASE_URL serving $MODEL (is this the right machine and port?)" >&2
    exit 1
fi

COMMON=(--model "$MODEL" --base-urls "$BASE_URL" --live-cache "$CACHE" --clusters "$CLUSTERS"
        --temperature "$TEMPERATURE" --workers "$WORKERS" --visible-reasoning)

# --- 0. search questions ----------------------------------------------------------
# Drawn once and shared by every v3 run, whatever the debate model.
if [[ -f "$CLUSTERS" ]]; then
    echo "[$(stamp)] step 0: $CLUSTERS exists, skipping"
else
    echo "[$(stamp)] step 0: drawing the search questions"
    "$PYTHON" scripts/sample_cluster_subsets.py --out "$CLUSTERS"
fi

if [[ ! -f "$CACHE" && -f "$OLD_CACHE" ]]; then
    echo "[$(stamp)] copying the v2 run's recordings to $CACHE as a starting cache"
    cp "$OLD_CACHE" "$CACHE.part" && mv "$CACHE.part" "$CACHE"    # never leave a half copy behind
fi

# --- 1. seeds -------------------------------------------------------------------
if [[ -f "$OUT/seeds.json" ]]; then
    echo "[$(stamp)] step 1: $OUT/seeds.json exists, skipping"
else
    echo "[$(stamp)] step 1: seed stage"
    "$PYTHON" scripts/program_seeds_v3.py --out "$OUT/seeds.json" "${COMMON[@]}" 2>&1 | tee -a "$OUT/seeds.log"
fi

# --- 2. search ------------------------------------------------------------------
if [[ -f "$RUN/archive.jsonl" ]]; then
    echo "[$(stamp)] step 2: resuming the search in $RUN"
    RESUME="--resume"
else
    echo "[$(stamp)] step 2: starting the search in $RUN"
    RESUME=""
fi
"$PYTHON" scripts/evolve_program_clusters_v3.py --seeds "$OUT/seeds.json" --out "$RUN" \
    "${COMMON[@]}" --generations "$GENERATIONS" $RESUME $EXTRA 2>&1 | tee -a "$RUN/search.log"

# --- 3. champions ---------------------------------------------------------------
echo "[$(stamp)] step 3: champions and literature baselines on held-out questions"
"$PYTHON" scripts/evolve_program_clusters_v3.py --out "$RUN" --pick-champions "${COMMON[@]}" \
    2>&1 | tee -a "$RUN/champions.log"

echo "[$(stamp)] done: $RUN/summary.json, $RUN/champions.json"

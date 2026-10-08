#!/usr/bin/env bash
# The per-group program search, start to finish: seed stage, search, champions.
# Every step is resumable, so re-running this script after an interruption
# picks up where it stopped (recorded rounds, the seed set and the archive are
# all reused). Run it in a tmux session:
#
#     bash run_cluster_search.sh 2>&1 | tee outputs/cluster_search/pipeline.log
#
# Assumes the vLLM server is up on $BASE_URL (Model_hosting/deploy_qwen35_35b_fp8.sh)
# and that `python` is the vllm-updated environment.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${OUT:-outputs/cluster_search}"          # seeds, cache, logs
RUN="${RUN:-$OUT/run1}"                       # this search's archive and results
BASE_URL="${BASE_URL:-http://localhost:7472/v1}"
WORKERS="${WORKERS:-64}"
GENERATIONS="${GENERATIONS:-40}"
EXTRA="${EXTRA:-}"                            # any extra flags for the search, e.g. "--no-guide"

mkdir -p "$OUT" "$RUN"
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(stamp)] checking the debate server at $BASE_URL"
if ! curl -sf --max-time 10 "$BASE_URL/models" > /dev/null; then
    echo "no server answering at $BASE_URL; start it first" >&2
    exit 1
fi

# --- 1. seeds -------------------------------------------------------------------
if [[ -f "$OUT/seeds.json" ]]; then
    echo "[$(stamp)] step 1: $OUT/seeds.json exists, skipping the seed stage"
else
    echo "[$(stamp)] step 1: seed stage"
    python scripts/program_seeds.py --out "$OUT/seeds.json" --base-urls "$BASE_URL" \
        --workers "$WORKERS" 2>&1 | tee -a "$OUT/seeds.log"
fi

# --- 2. search ------------------------------------------------------------------
if [[ -f "$RUN/archive.jsonl" ]]; then
    echo "[$(stamp)] step 2: resuming the search in $RUN"
    RESUME="--resume"
else
    echo "[$(stamp)] step 2: starting the search in $RUN"
    RESUME=""
fi
python scripts/evolve_program_clusters.py --seeds "$OUT/seeds.json" --out "$RUN" \
    --base-urls "$BASE_URL" --workers "$WORKERS" --generations "$GENERATIONS" \
    $RESUME $EXTRA 2>&1 | tee -a "$RUN/search.log"

# --- 3. champions ---------------------------------------------------------------
echo "[$(stamp)] step 3: champion picking on held-out questions"
python scripts/evolve_program_clusters.py --out "$RUN" --pick-champions \
    --base-urls "$BASE_URL" --workers "$WORKERS" 2>&1 | tee -a "$RUN/champions.log"

echo "[$(stamp)] done: $RUN/summary.json, $RUN/champions.json"

#!/usr/bin/env bash
# Our Direct CoT and Self-Refine against the external ones, before a cluster search (qwen9b run2, gptoss run4):
#   1. the external baselines on the test split, and 2. on the search questions (with the dev split);
#      both are already recorded for qwen9b, so they are only scored (pipeline_cluster_setup.sh). HLE
#      (2026-10-07): neither; the comparison runs on the first 25 search questions of each group (100),
#      and step 2 runs the external baselines on those alone, one run each
#   then our Direct CoT (direct_high) and our Self-Refine (self_refine_high) run on those questions
#   as many times as the external ones (3; HLE: 1), at replicates 0, 1, ... (the search's generation 0
#   reuses replicates 0 and 1 and runs the ones missing), and a low-effort solver runs once, for
#   the token ratio; they are compared with the external Direct CoT and Self-Refine
#   (scripts/compare_external_baselines.py) -> <run>/external_baselines.md and .json
# It ends there. Read the report; if ours match, start the search with run_pipeline_cluster.sh, which
# refuses to start without a passed comparison. Exit code 1 if the comparison fails or does not finish.
# Resumable: run it again after an interruption and finished debates are kept.
#
#     mkdir -p outputs/pipeline_cluster_qwen9b/run2
#     bash run_compare_external_cluster.sh qwen9b 2>&1 | tee -a outputs/pipeline_cluster_qwen9b/run2/compare.log
set -euo pipefail
cd "$(dirname "$0")"
source pipeline_cluster_setup.sh

if (( ! COMPARE_EXTERNAL )); then
    echo "the family $FAMILY has no comparison with the external baselines on $DATASET (set for qwen9b, qwen4b and" \
         "gptoss SuperGPQA, and every family on MATH and HLE)" >&2
    exit 1
fi
CHECK="$RUN/external_baselines"
echo "[$(stamp)] our Direct CoT and Self-Refine against the external ones on $CMP_Q -> $CHECK.md"
if ! "$PYTHON" scripts/compare_external_baselines.py --questions "$CMP_Q" --out "$CHECK" \
        --external-direct "$BRES/direct_${BNAME_CMP}_rec_k$CMP_K.json" \
        --external-direct-raw "$BRES/direct_${BNAME_CMP}.jsonl" \
        --external-selfrefine "$BRES/selfrefine_${BNAME_CMP}_rec_k$CMP_K.json" \
        --model "$MODEL" --base-urls "$BASE_URL" --live-cache "$CACHE" --workers "$WORKERS" "${EXECUTOR[@]}"; then
    echo "[$(stamp)] the comparison FAILED (or did not finish): see $CHECK.md. Do not start the search." >&2
    exit 1
fi
echo "[$(stamp)] ours match the external baselines: see $CHECK.md. To start the search:" \
     "bash run_pipeline_cluster.sh $FAMILY $DATASET"

#!/usr/bin/env bash
# Recover the results of the 15-hour Qwen3.5-27B run, then measure the
# reference programs it never got to.
#
# Background: that run finished its search and printed its results, but crashed
# in the reporting phase (a reference program still used the eliminator, which
# --no-eliminator had removed) before writing any output file. The search
# itself is intact: all 62k rounds are in the cache, and the search is
# deterministic given the seed, so step 1 replays it for free.
#
#   step 1  replay the whole search from cache and write the output files.
#           No model calls. Minutes.
#   step 2  the same run again, with the reference programs enabled, on 300 dev
#           questions at the fresh replicate. About 27.6k calls on all 1,515
#           would be ~22 h at the 0.35 calls/s this server gave; 300 questions
#           is ~4 h, and every program stays paired on the same questions.
#
# Step 2 needs the model served on :7472. Run it in tmux.
set -euo pipefail
cd "$(dirname "$0")"
touch outputs/empty.jsonl

CACHE=outputs/program_live_rounds_cache_qwen27b.jsonl

# every setting the original run used, so the search reproduces exactly
COMMON=(--evolve --live
        --model Qwen/Qwen3.5-27B
        --no-eliminator --digest-head 300 --digest-tail 900
        --min-coverage 0
        --generations 40 --population 16 --offspring 4
        --novelty-budget 300 --recheck-top 3
        --cache outputs/empty.jsonl --treegrow-cache outputs/empty.jsonl
        --live-cache "$CACHE")

RESULTS=outputs/program_mcq_evolved_live_qwen27b.json
WITH_BASE=outputs/program_mcq_evolved_live_qwen27b_with_baselines.json

# --- step 1: recover, no model calls ---------------------------------------
echo "[$(date '+%F %T')] step 1/2: replaying the search from cache (no model calls)"
python scripts/evolve_program_mcq.py "${COMMON[@]}" \
  --no-recheck-baselines \
  --evolved-out "$RESULTS" \
  --save-all outputs/program_mcq_all_live_qwen27b.jsonl

if [ ! -s "$RESULTS" ]; then
  echo "step 1 produced no results file; stopping before spending calls." >&2
  exit 1
fi
echo "[$(date '+%F %T')] step 1 done -> $RESULTS"

# --- step 2: the reference programs, fresh, paired on 300 dev questions -----
echo "[$(date '+%F %T')] step 2/2: reference programs on 300 dev questions (~4 h)"
python scripts/evolve_program_mcq.py "${COMMON[@]}" \
  --recheck-n 300 \
  --evolved-out "$WITH_BASE" \
  --save-all outputs/program_mcq_all_live_qwen27b_with_baselines.jsonl

echo "[$(date '+%F %T')] done"
echo "  recovered results:      $RESULTS"
echo "  results + baselines:    $WITH_BASE"
echo "  round cache:            $CACHE"

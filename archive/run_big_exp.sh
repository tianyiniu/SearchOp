#!/usr/bin/env bash
# Live program evolution on Qwen/Qwen3.5-27B (served on :7472), in two steps.
#
# Everything here is 27B-specific. The two Qwen3-14B recordings are replaced by
# an empty file: the round cache is keyed by (question, rounds, replicate) and
# NOT by model, so loading them would silently replay the old model's answers.
#
# Settings that differ from the 14B runs:
#   --no-eliminator            that persona refuses to answer, and its survivor
#                              list reached the next speaker only 2% of the time
#   --digest-head/-tail        78% of responses run past 700 chars and the
#                              committed conclusion sits at ~99% of the text, so
#                              a head-only window kept the setup and dropped the
#                              payoff. 300+900 keeps both ends. This also changes
#                              the round cache key, so these recordings can never
#                              be mistaken for the 700/0 ones.
#   --min-coverage 0           nothing is cached at the start, so the prescreen
#                              must not require prior coverage
#
# Before the first run:
#   python scripts/probe_env.py --model Qwen/Qwen3.5-27B \
#       --live-cache outputs/program_live_rounds_cache_qwen27b.jsonl --skip-recordings
#
# Run it in tmux; both steps are long.
set -euo pipefail
cd "$(dirname "$0")"
touch outputs/empty.jsonl

# _v2: recordings made by the v2 pipeline (careful-reasoning prompts, 6144-token
# replies, a summary call per persona whose text is the debate history). Keyed
# v=2, so the 14B and earlier 27B recordings can never be mistaken for them.
# The v2 cache started by run_v2_dev_overnight.sh is reused here as-is.
CACHE=outputs/program_live_rounds_cache_qwen27b_v2.jsonl
WARMUP_DONE=outputs/warmup_qwen27b_v2.json

COMMON=(--evolve --live
        --model Qwen/Qwen3.5-27B
        --no-eliminator --digest-head 300 --digest-tail 900
        --v2
        --min-coverage 0
        --cache outputs/empty.jsonl --treegrow-cache outputs/empty.jsonl
        --live-cache "$CACHE")

# --- step 1: warm the cache ------------------------------------------------
# With a cold cache the per-candidate novelty budget (300 calls) covers only 18
# of the 1,515 training questions, so every program scores near zero and
# selection is decided by noise. This step runs zero generations: it scores the
# seed programs with an effectively unlimited budget, which records the shared
# plan rounds (and every prefix of them) for both program families, then
# evaluates the best five on the dev half. Afterwards a candidate's 300 calls
# pay only for where it DIVERGES from the seeds -- on the 14B run, 79% of
# questions never left the plan at all.
#
# Cost: roughly 60-90k calls, depending on how much the seeds share prefixes.
# It is not extra work; it is the main run's first step done properly.
if [ -f "$WARMUP_DONE" ]; then
  echo "[$(date '+%F %T')] warm-up already done ($WARMUP_DONE exists, $(wc -l < "$CACHE") rounds cached); skipping"
else
  echo "[$(date '+%F %T')] step 1/2: warming the cache (seed programs, no generations)"
  python scripts/evolve_program_mcq.py "${COMMON[@]}" \
    --generations 0 \
    --novelty-budget 100000 \
    --recheck-top 0 --no-recheck-baselines \
    --evolved-out "$WARMUP_DONE"
  echo "[$(date '+%F %T')] warm-up done: $(wc -l < "$CACHE") rounds cached"
fi

# --- step 2: the search ----------------------------------------------------
echo "[$(date '+%F %T')] step 2/2: evolution (40 generations)"
python scripts/evolve_program_mcq.py "${COMMON[@]}" \
  --generations 40 --population 16 --offspring 4 \
  --novelty-budget 300 --recheck-top 3 \
  --evolved-out outputs/program_mcq_evolved_live_qwen27b_v2.json \
  --save-all outputs/program_mcq_all_live_qwen27b_v2.jsonl

echo "[$(date '+%F %T')] done"
echo "  results:      outputs/program_mcq_evolved_live_qwen27b_v2.json"
echo "  all programs: outputs/program_mcq_all_live_qwen27b_v2.jsonl"
echo "  round cache:  $CACHE"

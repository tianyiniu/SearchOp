#!/usr/bin/env bash
# Experiment 2 (FRAMES search programs), end to end, unattended.
#
# Stages: smoke test (5 questions) -> v0 baseline (all 266) -> evolution
# (train half, raced) -> winner scored on the untouched test half.
# A failed stage stops the run; everything is logged with progress lines.
#
# Searches AND page fetches go to live Serper (paid). Every external call is
# cached on disk by exact input (outputs/program_frames_callcache.jsonl), so
# nothing is ever paid for twice, across stages and across reruns.
#
# Needs before launch:
#   - vLLM serving Qwen/Qwen3-14B on :7472
#   - SERPER_API_KEY and OPENAI_API_KEY in the repo's .env (verified working)
#     or exported in the shell
#
# Run inside tmux:
#   tmux new -s frames
#   ./run_program_frames_overnight.sh
#   # detach with Ctrl-b d; reattach later with: tmux attach -t frames
#   # per-stage logs also land in outputs/overnight_<timestamp>/

set -euo pipefail
cd "$(dirname "$0")"
PY=/nas-ssd2/tianyin4/cache/venvs/vllm-host/bin/python3
LOG_DIR="outputs/overnight_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"
export TQDM_MININTERVAL=30          # progress lines every ~30s in the logs

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/run.log"; }

log "stage 0: smoke test (5 questions)"
$PY scripts/evolve_program_frames.py --v0 --limit 5 2>&1 | tee "$LOG_DIR/0_smoke.log"

log "stage 1: v0 baseline on all 266"
$PY scripts/evolve_program_frames.py --v0 2>&1 | tee "$LOG_DIR/1_v0.log"

log "stage 2: evolution (train half, raced)"
$PY scripts/evolve_program_frames.py --evolve 2>&1 | tee "$LOG_DIR/2_evolve.log"

log "stage 3: winner on the held-out test half"
$PY scripts/evolve_program_frames.py --test outputs/program_frames_evolved.json \
    2>&1 | tee "$LOG_DIR/3_test.log"

log "done"
log "v0 records:      outputs/search_program_v0_cache.jsonl"
log "top programs:    outputs/program_frames_evolved.json"
log "test records:    outputs/search_program_evolved_cache.jsonl"
log "call cache:      outputs/program_frames_callcache.jsonl (reused by reruns)"
log "logs:            $LOG_DIR/"

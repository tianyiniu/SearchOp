#!/usr/bin/env bash
# v2 pipeline, pass@1 on the FULL dev split (1,515 questions), unattended.
#
# v2 = careful-reasoning persona prompts, a 6,144-token reply budget so replies
# finish, and a paired summary call per persona whose text is what later
# personas read (the full reply stays in the cache). Thinking stays off in
# every request. Recordings go to a NEW cache keyed v=2, so nothing here can
# replay or pollute the earlier runs.
#
# Three steps, cheapest first, each skipped if its output already exists, all
# sharing one cache. Vote variants read the same transcripts, so they are free.
#   step 1  solver -> critic            (~3k reasoning + ~3k summary calls)
#   step 2  fixed 8-call recipe         (~12k + ~12k)
#   step 3  program B + evolved winner  (~16k + ~16k)
# At the ~0.35 calls/s the old server gave, step 1 is ~5 h and steps 2-3 ~20 h
# each; run the later steps on other nights, the script resumes past done ones.
#
# pass@1 is "avg_at_k" (k=1) in each --out file; per-question outcomes and
# letters are there too, in the same dev order as every earlier passk file.
#
# Needs: vLLM serving Qwen/Qwen3.5-27B on :7472 with thinking off by default
#   (--default-chat-template-kwargs '{"enable_thinking": false}'); the scripts
#   also ask for thinking off on every request.
# Run inside tmux:
#   tmux new -s v2dev
#   ./run_v2_dev_overnight.sh
#   WORKERS=96 ./run_v2_dev_overnight.sh      # more concurrency if the server has KV room
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$(command -v python || echo /nas-ssd2/tianyin4/cache/venvs/vllm-host/bin/python)}
export TQDM_MININTERVAL=30
touch outputs/empty.jsonl

MODEL=Qwen/Qwen3.5-27B
CACHE=outputs/program_live_rounds_cache_qwen27b_v2.jsonl
EVOLVED=outputs/program_mcq_evolved_live_qwen27b_with_baselines.json
WORKERS=${WORKERS:-48}
OUT1=outputs/passk_v2_dev_solver_critic.json
OUT2=outputs/passk_v2_dev_program_fixed.json
OUT3=outputs/passk_v2_dev_program_b.json
LOG_DIR="outputs/v2_dev_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/run.log"; }

COMMON=(--live --model "$MODEL" --v2
        --no-eliminator --digest-head 300 --digest-tail 900
        --k 1 --reps 0 --workers "$WORKERS"
        --cache outputs/empty.jsonl --treegrow-cache outputs/empty.jsonl
        --live-cache "$CACHE")

# --- preflight -------------------------------------------------------------
log "preflight"
[ -s "$EVOLVED" ] || { log "$EVOLVED missing (needed for step 3's evolved winner)"; exit 1; }
[ -f "$CACHE.lock" ] && { log "$CACHE.lock exists: another run holds the cache. Stop it or delete the lock."; exit 1; }
$PY - <<'PYEOF' || { log "vLLM on :7472 not answering; start the server first"; exit 1; }
import socket, sys
s = socket.socket(); s.settimeout(3)
sys.exit(0 if s.connect_ex(("127.0.0.1", 7472)) == 0 else 1)
PYEOF
log "vLLM :7472 ok; interpreter $PY; workers $WORKERS; cache $CACHE"

# --- step 1: solver -> critic ----------------------------------------------
if [ -s "$OUT1" ]; then
  log "step 1/3: $OUT1 exists; skipping"
else
  log "step 1/3: solver_critic (+ vote read), full dev, fresh replicate 0"
  $PY scripts/eval_passk_programs.py "${COMMON[@]}" \
      --refs solver_critic,solver_critic_vote --out "$OUT1" 2>&1 | tee "$LOG_DIR/1_solver_critic.log"
  log "step 1 done -> $OUT1"
fi

# --- step 2: the fixed 8-call recipe ---------------------------------------
if [ -s "$OUT2" ]; then
  log "step 2/3: $OUT2 exists; skipping"
else
  log "step 2/3: program_fixed (+ vote read), full dev"
  $PY scripts/eval_passk_programs.py "${COMMON[@]}" \
      --refs program_fixed,program_fixed_vote --out "$OUT2" 2>&1 | tee "$LOG_DIR/2_program_fixed.log"
  log "step 2 done -> $OUT2"
fi

# --- step 3: program B and the evolved winner --------------------------------
if [ -s "$OUT3" ]; then
  log "step 3/3: $OUT3 exists; skipping"
else
  log "step 3/3: program_b (+ vote read) and evolved_1, full dev"
  $PY scripts/eval_passk_programs.py "${COMMON[@]}" \
      --refs program_b,program_b_vote --evolved-out "$EVOLVED" --top 1 \
      --out "$OUT3" 2>&1 | tee "$LOG_DIR/3_program_b.log"
  log "step 3 done -> $OUT3"
fi

log "done"
log "  solver_critic:  $OUT1"
log "  program_fixed:  $OUT2"
log "  program_b:      $OUT3"
log "  v2 cache:       $CACHE"
log "  logs:           $LOG_DIR/"

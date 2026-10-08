#!/usr/bin/env bash
# The commit-follow-up experiment on Qwen3.5-27B, end to end, unattended.
#
# Background: ~19% of solver replies (32% of verifier replies) ran out of output
# tokens before their ANSWER line and counted as silence. The follow-up sends
# one short continuation asking such a reply to commit. This script measures
# what that, plus a plurality read-off, is worth on the same 300 dev questions
# the earlier fresh rechecks used.
#
#   step 1  back-fill commits into the existing recordings -> a new cache file
#           under the follow-up key. ~15k short calls. The source is untouched.
#   step 2  replay replicates 0,1 on the repaired cache. Nearly free. Only the
#           round-1 votes and the read-off are exact here (later personas were
#           generated before the commits existed), so this is the quick read.
#   step 3  fresh replicates 2,3 with the follow-up on from the first round.
#           ~12k calls. This is the number to quote.
#
# Each step is skipped if its output already exists, so a rerun resumes. Every
# script prints tqdm progress; per-step logs land in outputs/commit_fix_<stamp>/.
#
# Needs: vLLM serving Qwen/Qwen3.5-27B on :7472. Run inside tmux:
#   tmux new -s commitfix
#   ./run_commit_fix_overnight.sh
set -euo pipefail
cd "$(dirname "$0")"
# interpreter: $PY if set, else `python` on PATH, else the vllm-host venv
PY=${PY:-$(command -v python || echo /nas-ssd2/tianyin4/cache/venvs/vllm-host/bin/python)}
export TQDM_MININTERVAL=30          # progress lines every ~30s in the logs
touch outputs/empty.jsonl

MODEL=Qwen/Qwen3.5-27B
SRC=outputs/program_live_rounds_cache_qwen27b.jsonl
CACHE=outputs/program_live_rounds_cache_qwen27b_commit.jsonl
REPLAY_OUT=outputs/passk_qwen27b_commit_replay.json
FRESH_OUT=outputs/passk_qwen27b_commit_fresh.json
LOG_DIR="outputs/commit_fix_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/run.log"; }

REFS=program_b,program_b_vote,program_fixed,program_fixed_vote,solver_critic,solver_critic_vote
COMMON=(--live --model "$MODEL"
        --no-eliminator --digest-head 300 --digest-tail 900 --commit-followup
        --refs "$REFS" --n-dev 300
        --cache outputs/empty.jsonl --treegrow-cache outputs/empty.jsonl
        --live-cache "$CACHE")

# --- preflight -------------------------------------------------------------
log "preflight"
[ -s "$SRC" ] || { log "source cache $SRC missing"; exit 1; }
[ -f "$CACHE.lock" ] && { log "$CACHE.lock exists: another run holds the new cache. Stop it or delete the lock."; exit 1; }
$PY - <<'PYEOF' || { log "vLLM on :7472 not answering; start the server first"; exit 1; }
import socket, sys
s = socket.socket(); s.settimeout(3)
sys.exit(0 if s.connect_ex(("127.0.0.1", 7472)) == 0 else 1)
PYEOF
log "vLLM :7472 ok; interpreter $($PY -c 'import sys; print(sys.executable)')"

# --- step 1: repair the recordings ------------------------------------------
if [ -s "$CACHE" ]; then
  log "step 1/3: repaired cache exists ($(wc -l < "$CACHE") rounds); skipping"
else
  log "step 1/3: back-filling commits into $SRC -> $CACHE"
  $PY scripts/repair_truncated_rounds.py --model "$MODEL" --src "$SRC" --dst "$CACHE.partial" \
      --workers 32 2>&1 | tee "$LOG_DIR/1_repair.log"
  mv "$CACHE.partial" "$CACHE"        # atomic: a crash mid-write never leaves a half cache
  log "step 1 done: $(wc -l < "$CACHE") rounds"
fi

# --- step 2: quick read on recorded replicates 0,1 --------------------------
if [ -s "$REPLAY_OUT" ]; then
  log "step 2/3: $REPLAY_OUT exists; skipping"
else
  log "step 2/3: replicates 0,1 on the repaired cache (nearly free)"
  $PY scripts/eval_passk_programs.py "${COMMON[@]}" --k 2 --reps 0,1 \
      --out "$REPLAY_OUT" 2>&1 | tee "$LOG_DIR/2_replay.log"
  log "step 2 done -> $REPLAY_OUT"
fi

# --- step 3: fresh replicates 2,3 with the follow-up on throughout ----------
if [ -s "$FRESH_OUT" ]; then
  log "step 3/3: $FRESH_OUT exists; skipping"
else
  log "step 3/3: fresh replicates 2,3 (~12k calls)"
  $PY scripts/eval_passk_programs.py "${COMMON[@]}" --k 2 --reps 2,3 \
      --out "$FRESH_OUT" 2>&1 | tee "$LOG_DIR/3_fresh.log"
  log "step 3 done -> $FRESH_OUT"
fi

log "done"
log "  quick read (recorded reps 0,1): $REPLAY_OUT"
log "  clean number (fresh reps 2,3):  $FRESH_OUT"
log "  repaired cache:                 $CACHE"
log "  logs:                           $LOG_DIR/"

#!/usr/bin/env bash
# Round 2 of the commit-follow-up experiment: a better nudge, and the FULL dev
# split (1,515 questions), unattended.
#
# Round 1 (run_commit_fix_overnight.sh) showed the first nudge landed a letter
# on only 32% of cut-off replies: the model began a prose summary and was cut
# off again at 48 tokens. The nudge now puts the letter first, allows 160
# tokens, and a lenient parser reads a letter out of a prose commit reply.
#
#   step 1  re-nudge: every failed follow-up in the round-1 cache is retried
#           with the new nudge (old ones that a parser can read are fixed for
#           free). ~15k short calls; at the ~1 reply/s seen in round 1, ~4 h.
#   step 2  full dev, recorded replicates 0,1, all reference programs plus the
#           evolved winner, on the re-nudged cache. Nearly free.
#   step 3  full dev, ONE fresh replicate (4), follow-up on from the first
#           round, for the programs in FRESH_REFS. Default is the 2-call
#           solver->critic pair (~3k calls + follow-ups, ~2-3 h). Adding
#           program_b costs ~15k more calls (~12 h) -- run that another night:
#             FRESH_REFS=program_b,program_b_vote ./run_commit_fix_fulldev_overnight.sh
#
# Each step is skipped if its output exists, so a rerun resumes. Every script
# prints tqdm progress; per-step logs land in outputs/commit_fix2_<stamp>/.
#
# Needs: vLLM serving Qwen/Qwen3.5-27B on :7472, and round 1's repaired cache.
# Run inside tmux:
#   tmux new -s commitfix2
#   ./run_commit_fix_fulldev_overnight.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$(command -v python || echo /nas-ssd2/tianyin4/cache/venvs/vllm-host/bin/python)}
export TQDM_MININTERVAL=30
touch outputs/empty.jsonl

MODEL=Qwen/Qwen3.5-27B
SRC=outputs/program_live_rounds_cache_qwen27b_commit.jsonl        # round 1's repaired cache
CACHE=outputs/program_live_rounds_cache_qwen27b_commit2.jsonl     # re-nudged; new rounds go here
EVOLVED=outputs/program_mcq_evolved_live_qwen27b_with_baselines.json
REPLAY_OUT=outputs/passk_qwen27b_commit2_fulldev_replay.json
FRESH_OUT=outputs/passk_qwen27b_commit2_fulldev_fresh.json
FRESH_REFS=${FRESH_REFS:-solver_critic,solver_critic_vote}
LOG_DIR="outputs/commit_fix2_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/run.log"; }

ALL_REFS=program_b,program_b_vote,program_fixed,program_fixed_vote,solver_critic,solver_critic_vote
COMMON=(--live --model "$MODEL"
        --no-eliminator --digest-head 300 --digest-tail 900 --commit-followup
        --cache outputs/empty.jsonl --treegrow-cache outputs/empty.jsonl
        --live-cache "$CACHE")

# --- preflight -------------------------------------------------------------
log "preflight"
[ -s "$SRC" ] || { log "round-1 cache $SRC missing; run run_commit_fix_overnight.sh first"; exit 1; }
[ -s "$EVOLVED" ] || { log "$EVOLVED missing"; exit 1; }
[ -f "$CACHE.lock" ] && { log "$CACHE.lock exists: another run holds the cache. Stop it or delete the lock."; exit 1; }
$PY - <<'PYEOF' || { log "vLLM on :7472 not answering; start the server first"; exit 1; }
import socket, sys
s = socket.socket(); s.settimeout(3)
sys.exit(0 if s.connect_ex(("127.0.0.1", 7472)) == 0 else 1)
PYEOF
log "vLLM :7472 ok; interpreter $PY; fresh refs: $FRESH_REFS"

# --- step 1: re-nudge the failed follow-ups ---------------------------------
if [ -s "$CACHE" ]; then
  log "step 1/3: re-nudged cache exists ($(wc -l < "$CACHE") rounds); skipping"
else
  log "step 1/3: re-nudging failed follow-ups $SRC -> $CACHE"
  $PY scripts/repair_truncated_rounds.py --renudge --model "$MODEL" --src "$SRC" \
      --dst "$CACHE.partial" --workers 32 2>&1 | tee "$LOG_DIR/1_renudge.log"
  mv "$CACHE.partial" "$CACHE"
  log "step 1 done: $(wc -l < "$CACHE") rounds"
fi

# --- step 2: full dev on recorded replicates 0,1 ----------------------------
if [ -s "$REPLAY_OUT" ]; then
  log "step 2/3: $REPLAY_OUT exists; skipping"
else
  log "step 2/3: full dev (1,515 q), replicates 0,1, references + evolved winner (nearly free)"
  $PY scripts/eval_passk_programs.py "${COMMON[@]}" \
      --evolved-out "$EVOLVED" --top 1 --refs "$ALL_REFS" \
      --k 2 --reps 0,1 --out "$REPLAY_OUT" 2>&1 | tee "$LOG_DIR/2_replay_fulldev.log"
  log "step 2 done -> $REPLAY_OUT"
fi

# --- step 3: full dev, one fresh replicate ----------------------------------
if [ -s "$FRESH_OUT" ]; then
  log "step 3/3: $FRESH_OUT exists; skipping"
else
  log "step 3/3: full dev, fresh replicate 4, programs: $FRESH_REFS"
  $PY scripts/eval_passk_programs.py "${COMMON[@]}" \
      --refs "$FRESH_REFS" --k 1 --reps 4 --out "$FRESH_OUT" 2>&1 | tee "$LOG_DIR/3_fresh_fulldev.log"
  log "step 3 done -> $FRESH_OUT"
fi

log "done"
log "  full dev, recorded reps 0,1 (quick read): $REPLAY_OUT"
log "  full dev, fresh rep 4 ($FRESH_REFS):       $FRESH_OUT"
log "  re-nudged cache:                            $CACHE"
log "  logs:                                       $LOG_DIR/"

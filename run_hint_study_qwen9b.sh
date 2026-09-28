#!/usr/bin/env bash
# The hint-sheet study with Qwen3.5-9B as the small model. Same steps as run_hint_study.sh;
# the differences are the model, its own output folder, and the shared hint sheets: the big
# model's sheets are read from (and, for questions this model fails that gpt-oss did not,
# added to) $TEACHER_DIR, so each sheet is written once. Run inside tmux.
#
#   bash run_hint_study_qwen9b.sh                # everything, in order
#   STEPS="2 3" bash run_hint_study_qwen9b.sh    # only some steps
#
# Needs: a Qwen/Qwen3.5-9B vLLM server (Model_hosting/deploy_qwen35_9b.sh, port 7472). The 9B
# is hosted on another machine, so either run this script on that machine (the default
# endpoint, localhost:7472, is then right) or run it here with
#   ENDPOINTS=http://<that machine>:7472 bash run_hint_study_qwen9b.sh
# The script refuses to start unless the endpoint serves $MODEL. OPENAI_API_KEY in .env is
# needed for the big model. "Light thinking" for Qwen means thinking off.
#
# Running alongside the gpt-oss run: step 2 only touches this model's own folder, so it can run
# at any time. Step 3 appends to the shared sheet file in $TEACHER_DIR, so run it after the
# gpt-oss run has finished its own step 3 (STEPS="2" now, STEPS="3 4 5 7 8" later).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
ENDPOINTS="${ENDPOINTS:-http://localhost:7472}"
OUT_DIR="${OUT_DIR:-outputs/hint_study_qwen35_9b}"
TEACHER_DIR="${TEACHER_DIR:-outputs/hint_study}"        # the gpt-oss run's folder: sheets are shared
WORKERS="${WORKERS:-32}"
STEPS="${STEPS:-2 3 4 5 7 8}"                           # step 1 (the draw) is shared with the gpt-oss run
H=scripts/hint_study
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

FIRST="${ENDPOINTS%%,*}"
if ! curl -sf --max-time 10 "$FIRST/v1/models" | grep -q "\"$MODEL\""; then
    echo "no server at $FIRST serving $MODEL" >&2; exit 1
fi
[[ -f datasets/hint_study_train.json ]] || { echo "run scripts/hint_study/draw_questions.py first" >&2; exit 1; }
mkdir -p "$OUT_DIR"

STUDENT=(--model "$MODEL" --endpoints "$ENDPOINTS" --workers "$WORKERS" --out-dir "$OUT_DIR" --teacher-dir "$TEACHER_DIR")
DIRS=(--out-dir "$OUT_DIR" --teacher-dir "$TEACHER_DIR")

for step in $STEPS; do
  case "$step" in
    2) echo "[$(stamp)] step 2: can $MODEL do them alone? (5 tries, thinking on)"
       "$PYTHON" $H/unaided.py --split both "${STUDENT[@]}" ;;
    3) echo "[$(stamp)] step 3: hint sheets (shared with $TEACHER_DIR), then this model's give-away check"
       "$PYTHON" $H/hint_sheets.py --split both "${DIRS[@]}"
       "$PYTHON" $H/leak_check.py --split both "${STUDENT[@]}"
       "$PYTHON" $H/hint_sheets.py --split both --strict "${DIRS[@]}"
       "$PYTHON" $H/leak_check.py --split both "${STUDENT[@]}" ;;
    4) echo "[$(stamp)] step 4: keep the middle questions and trim the sheets"
       "$PYTHON" $H/gate_with_sheet.py --split both --trim "${STUDENT[@]}" ;;
    5) echo "[$(stamp)] step 5: hints needed by each way of working (plus the full plain curve)"
       "$PYTHON" $H/hints_needed.py --split both --full-curve "${STUDENT[@]}" ;;
    7) echo "[$(stamp)] step 7: which facts the model has"
       "$PYTHON" $H/fact_probe.py --split both "${STUDENT[@]}" ;;
    8) echo "[$(stamp)] report"
       "$PYTHON" $H/report.py "${DIRS[@]}" ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done
echo "[$(stamp)] done"

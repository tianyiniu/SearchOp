#!/usr/bin/env bash
# The hint-sheet study (backward_scaffold_experiment.md), steps 1 to 5 and 7, on the
# 100 + 100 drawn questions with K = 5 tries. Run inside tmux. Every step resumes from
# its cache, so the script can be rerun after an interruption.
#
#   bash run_hint_study.sh                 # everything below, in order
#   STEPS="2 3" bash run_hint_study.sh     # only some steps
#
# Needs: a gpt-oss-20b vLLM server (Model_hosting/deploy_gpt_oss_20b.sh) at $ENDPOINTS,
# and OPENAI_API_KEY in .env for the big model.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python}"
ENDPOINTS="${ENDPOINTS:-http://localhost:7472}"
WORKERS="${WORKERS:-32}"
STEPS="${STEPS:-1 2 3 4 5 7 8}"
H=scripts/hint_study
stamp() { date "+%Y-%m-%d %H:%M:%S"; }

for step in $STEPS; do
  case "$step" in
    1) echo "[$(stamp)] step 1: draw 100 + 100 questions from the labelled pool"
       "$PYTHON" $H/draw_questions.py ;;
    2) echo "[$(stamp)] step 2: can the small model do them alone? (5 tries, full thinking)"
       "$PYTHON" $H/unaided.py --split both --endpoints "$ENDPOINTS" --workers "$WORKERS" ;;
    3) echo "[$(stamp)] step 3: hint sheets from the big model, then the give-away check"
       "$PYTHON" $H/hint_sheets.py --split both
       "$PYTHON" $H/leak_check.py --split both --endpoints "$ENDPOINTS" --workers "$WORKERS"
       "$PYTHON" $H/hint_sheets.py --split both --strict
       "$PYTHON" $H/leak_check.py --split both --endpoints "$ENDPOINTS" --workers "$WORKERS" ;;
    4) echo "[$(stamp)] step 4: keep the middle questions and trim the sheets"
       "$PYTHON" $H/gate_with_sheet.py --split both --trim --endpoints "$ENDPOINTS" --workers "$WORKERS" ;;
    5) echo "[$(stamp)] step 5: hints needed by each way of working (plus the full plain curve)"
       "$PYTHON" $H/hints_needed.py --split both --full-curve --endpoints "$ENDPOINTS" --workers "$WORKERS" ;;
    7) echo "[$(stamp)] step 7: which facts the model has"
       "$PYTHON" $H/fact_probe.py --split both --endpoints "$ENDPOINTS" --workers "$WORKERS" ;;
    8) echo "[$(stamp)] report"
       "$PYTHON" $H/report.py ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done
echo "[$(stamp)] done"

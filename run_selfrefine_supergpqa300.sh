#!/usr/bin/env bash
# The external Self-Refine alone on the 300 SuperGPQA test questions (datasets/supergpqa_600_test.json):
# up to 2 feedback -> refine rounds (baselines/run_baselines.py's default since 2026-10-07, as the
# pipeline's self_refine_high), 3 runs per question, with recovery, then scored. The model's server
# must be up (Model_hosting/deploy_*.sh). Resumable: run the same command again after a stop.
#
#     bash run_selfrefine_supergpqa300.sh qwen35-9b
#     bash run_selfrefine_supergpqa300.sh gptoss-20b
#     bash run_selfrefine_supergpqa300.sh qwen35-4b
#
# Any further option goes to run_baselines.py, e.g. --port 7474 for a server on another port.
# Writes baselines/results/selfrefine_it2_<tag>_supergpqa_600_test_rec.jsonl and the table
# baselines/results/table_<tag>_supergpqa_600_test_rec.md (<tag>: qwen35_9b_think, qwen35_4b_think,
# gptoss20b_high_explain).
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:?the model: qwen35-9b, qwen35-4b or gptoss-20b}"
shift
exec /nas-ssd2/tianyin4/cache/venvs/vllm-updated/bin/python baselines/run_baselines.py --model "$MODEL" \
    --data datasets/supergpqa_600_test.json --methods selfrefine "$@"

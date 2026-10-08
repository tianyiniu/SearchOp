#!/usr/bin/env bash
# Qwen/Qwen3.5-9B on an A100 80GB (a rented GPU). The model, port, window, parsers and chat options are
# those of deploy_qwen35_9b.sh: the pipeline checks the model name and the 32,768-token window, and the
# window is part of every recording's key. Only the memory and placement settings differ:
#   - one GPU by default (GPU 0); a list such as 0,1 starts one copy per GPU;
#   - the weights go to the Hugging Face cache (~/.cache/huggingface, or $HF_HOME), not the NAS;
#   - 92% of the 80GB for the model and its cache (deploy_qwen35_9b.sh: 95% of a 96GB card), which
#     leaves about 6GB for the CUDA graphs of 512 running requests;
#   - up to 512 requests at once: the pipeline runs 128 debates, each with up to 4 speakers in flight.
#
#     bash Model_hosting/deploy_qwen35_9b_a100.sh          # GPU 0
#     bash Model_hosting/deploy_qwen35_9b_a100.sh 0,1      # GPUs 0 and 1
#
# The server is ready when http://localhost:7472/v1/models answers (the first start downloads about
# 19GB of weights).

set -euo pipefail

CUDA_DEVICES="${1:-0}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
echo "Qwen/Qwen3.5-9B: $N_GPUS copy/copies, one per GPU, on $CUDA_DEVICES"

HF_HUB_CACHE=/workspace/models FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" vllm serve "Qwen/Qwen3.5-9B" \
  --trust-remote-code --host localhost --port 7472 \
  --max-model-len 32768 \
  --data-parallel-size "$N_GPUS" \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --async-scheduling

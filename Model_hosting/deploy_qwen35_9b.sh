#!/usr/bin/env bash

set -euo pipefail

CUDA_DEVICES="${1:-5,6}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
echo "Qwen/Qwen3.5-9B: $N_GPUS replica(s), one per GPU, on $CUDA_DEVICES"

FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" vllm serve "Qwen/Qwen3.5-9B" \
  --trust-remote-code --host localhost --port 7472 \
  --download-dir /nas-ssd2/tianyin4/cache/pretrained_models \
  --max-model-len 32768 \
  --data-parallel-size "$N_GPUS" \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --async-scheduling

#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICES="${1:?Usage: $0 <cuda_visible_devices, e.g. 2 or 0,1,2,3>}"

TP_SIZE=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")

# CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" VLLM_USE_V1=1 vllm serve "Qwen/Qwen3-14B" \
#   --trust-remote-code \
#   --host localhost \
#   --port 7472 \
#   --download-dir /nas-ssd2/tianyin4/cache/pretrained_models \
#   --max-model-len 32768 \
#   --tensor-parallel-size "$TP_SIZE" \
#   --enable-auto-tool-choice \
#   --tool-call-parser hermes \
#   --gpu-memory-utilization 0.85 

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" VLLM_USE_V1=1 vllm serve "Qwen/Qwen3.5-27B" \
  --trust-remote-code \
  --host localhost \
  --port 7472 \
  --download-dir /nas-ssd2/tianyin4/cache/pretrained_models \
  --max-model-len 32768 \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --gpu-memory-utilization 0.9

# --default-chat-template-kwargs applies the switch to EVERY request (vllm
# merges server defaults first, a request-level chat_template_kwargs still
# overrides). It helps only if the server was rejecting our per-request kwarg.
# If Qwen3.5's chat template ignores enable_thinking instead, this changes
# nothing and the model keeps thinking. To tell the two apart:
#   python scripts/thinking_probe.py http://localhost:7472/v1 Qwen/Qwen3.5-27B
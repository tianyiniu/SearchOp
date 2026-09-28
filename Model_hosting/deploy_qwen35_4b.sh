#!/usr/bin/env bash
# Qwen3.5-35B-A3B-FP8 (MoE, ~3B active, ~36 GB of weights): one replica per
# card, no tensor parallel. Fits a 48 GB card with ~7-9 GB of KV cache; read
# "GPU KV cache size" / "maximum concurrency" from the startup log and, if the
# per-replica concurrency is under ~30, switch to pairs instead:
#   --data-parallel-size $((N_GPUS/2)) --tensor-parallel-size 2 --enable-expert-parallel
# Thinking is off by server default; tool calling stays on.
#
# --max-num-batched-tokens: this model has Mamba-style (gated DeltaNet) layers.
# With prefix caching on, vLLM uses its "align" cache mode and sizes the
# attention block to the Mamba page (2096 tokens); that block must fit in one
# prefill batch, and the default batch is 2048, so startup asserted. 4096
# clears it and makes prefill a little faster.
#
#   ./Model_hosting/deploy_qwen35_35b_fp8.sh              # all four cards
#   ./Model_hosting/deploy_qwen35_35b_fp8.sh 0,1,2,7      # any card list
echo "SET PORT!!! CURRENTLY SCRIPT USES PORT 7473"

set -euo pipefail

CUDA_DEVICES="${1:-0,1,2,3}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
echo "Qwen/Qwen3.5-4B: $N_GPUS replica(s), one per GPU, on $CUDA_DEVICES"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" vllm serve "Qwen/Qwen3.5-4B" \
  --trust-remote-code --host localhost --port 7473 \
  --download-dir /nas-ssd2/tianyin4/cache/pretrained_models \
  --max-model-len 32768 \
  --data-parallel-size "$N_GPUS" \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.9



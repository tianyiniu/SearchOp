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

# NCCL must match the torch build in the active venv.
# - vllm-host (torch 2.13 + CUDA 13): use the venv's NCCL. Do NOT preload the
#   CUDA 12 copy: it is 2.27.5 and this torch needs symbols it lacks
#   (ncclCommResume), so `import torch` fails with it.
# - vllm-updated (torch 2.10 + CUDA 12.8): the venv's libnccl.so.2 was
#   overwritten by nvidia-nccl-cu13 (2.28.9, CUDA 13). On a driver older than
#   CUDA 13 (e.g. the Ada servers) NCCL init fails with "CUDA driver version is
#   insufficient for CUDA runtime version". Preload the CUDA 12 copy instead.
VLLM_PY="$(dirname "$(command -v vllm)")/python"
TORCH_CUDA=$("$VLLM_PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)
echo "torch CUDA: ${TORCH_CUDA:-unknown}"
NCCL_CU12="/nas-ssd2/tianyin4/cache/nccl_cu12_2.27.5/pkg/nvidia/nccl/lib/libnccl.so.2"
if [[ "$TORCH_CUDA" == 12.* ]]; then
  if [[ -f "$NCCL_CU12" ]]; then
    export VLLM_NCCL_SO_PATH="$NCCL_CU12"
    export LD_PRELOAD="$NCCL_CU12${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "NCCL: $NCCL_CU12"
  else
    echo "warning: $NCCL_CU12 not found; using the venv's NCCL" >&2
  fi
fi

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



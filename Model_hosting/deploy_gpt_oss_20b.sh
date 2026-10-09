#!/usr/bin/env bash
# Usage:
#   ./Model_hosting/deploy_gpt_oss_20b.sh              # all GPUs on this machine, port 7472
#   ./Model_hosting/deploy_gpt_oss_20b.sh 0            # a single card
#   ./Model_hosting/deploy_gpt_oss_20b.sh 0,1 7473     # two cards, another port
set -euo pipefail

DEFAULT_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
CUDA_DEVICES="${1:-$DEFAULT_DEVICES}"
PORT="${2:-7472}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
echo "gpt-oss-20b: $N_GPUS copies, one per GPU, on $CUDA_DEVICES, port $PORT"

# One compile cache per GPU model.
GPU_TAG=$(nvidia-smi -i "${CUDA_DEVICES%%,*}" --query-gpu=name --format=csv,noheader | tr -c 'A-Za-z0-9\n' '_')
# This server's NAS; elsewhere (a rented GPU) vLLM's own cache and the Hugging Face cache.
DOWNLOAD=()
if [[ -d /nas-ssd2/tianyin4/cache ]]; then
  export VLLM_CACHE_ROOT="/nas-ssd2/tianyin4/cache/vllm_by_gpu/$GPU_TAG"
  echo "vLLM cache: $VLLM_CACHE_ROOT"
  DOWNLOAD=(--download-dir /nas-ssd2/tianyin4/cache/pretrained_models)
fi

# Busy-card check and NCCL settings.
nvidia-smi -i "$CUDA_DEVICES" --query-gpu=index,memory.used,memory.total --format=csv
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

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

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" exec vllm serve "openai/gpt-oss-20b" \
  --host localhost --port "$PORT" \
  "${DOWNLOAD[@]}" \
  --max-model-len 32768 \
  --data-parallel-size "$N_GPUS" \
  --enable-prefix-caching \
  --reasoning-parser openai_gptoss \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --async-scheduling

#!/usr/bin/env bash
set -euo pipefail
echo "USING PORT 7473!!!"

CUDA_DEVICES="${1:-0,1,2,3}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
echo "Qwen3.5-35B-A3B-FP8: $N_GPUS replica(s), one per GPU, on $CUDA_DEVICES"

GPU_TAG=$(nvidia-smi -i "${CUDA_DEVICES%%,*}" --query-gpu=name --format=csv,noheader | tr -c 'A-Za-z0-9\n' '_')
export VLLM_CACHE_ROOT="/nas-ssd2/tianyin4/cache/vllm_by_gpu/$GPU_TAG"
echo "vLLM cache: $VLLM_CACHE_ROOT"

nvidia-smi -i "$CUDA_DEVICES" --query-gpu=index,memory.used,memory.total --format=csv

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

NCCL_CU12="/nas-ssd2/tianyin4/cache/nccl_cu12_2.27.5/pkg/nvidia/nccl/lib/libnccl.so.2"
if [[ -f "$NCCL_CU12" ]]; then
  export VLLM_NCCL_SO_PATH="$NCCL_CU12"
  export LD_PRELOAD="$NCCL_CU12${LD_PRELOAD:+:$LD_PRELOAD}"
else
  echo "warning: $NCCL_CU12 not found; using the venv's NCCL" >&2
fi

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" vllm serve "Qwen/Qwen3.5-35B-A3B-FP8" \
  --trust-remote-code --host localhost --port 7473 \
  --download-dir /nas-ssd2/tianyin4/cache/pretrained_models \
  --max-model-len 32768 \
  --data-parallel-size "$N_GPUS" \
  --kv-cache-dtype fp8 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.9

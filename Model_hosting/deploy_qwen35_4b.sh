#!/usr/bin/env bash
# Qwen/Qwen3.5-4B, one copy per GPU (data parallel), on port 7473 (the port the pipeline and
# run_baselines.py use for qwen4b). The window (32,768 tokens), parsers and chat options are those of
# deploy_qwen35_9b.sh: the pipeline checks the model name and the window.
#
#     bash Model_hosting/deploy_qwen35_4b.sh             # GPUs 0,1,2,3
#     bash Model_hosting/deploy_qwen35_4b.sh 0,1         # any card list
#
# The weights (about 9 GB) go to this server's NAS cache if it exists, else to the Hugging Face
# cache (~/.cache/huggingface, or $HF_HOME). The server is ready when
# http://localhost:7473/v1/models answers.
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

echo "4B USES PORT 7473"
DOWNLOAD=()
[[ -d /nas-ssd2/tianyin4/cache/pretrained_models ]] && DOWNLOAD=(--download-dir /nas-ssd2/tianyin4/cache/pretrained_models)

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" vllm serve "Qwen/Qwen3.5-4B" \
  --trust-remote-code --host localhost --port 7473 \
  "${DOWNLOAD[@]}" \
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


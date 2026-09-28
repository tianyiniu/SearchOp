#!/usr/bin/env bash
# openai/gpt-oss-20b (MoE, 21B total, ~3.6B active, ~14 GB of MXFP4 weights):
# one replica per card, no tensor parallel. The weights fit in 16 GB, so even a
# half-free card leaves tens of GB of KV cache; the KV cache stays at its
# default dtype (no fp8 needed, and fp8 KV with this model's attention sinks is
# not supported on every backend).
#
# Format. The model only works in OpenAI's "harmony" format: a system message
# that carries "Reasoning: low|medium|high", then the conversation, and replies
# split into channels ("analysis" = the chain of thought, "final" = the answer).
# vLLM builds harmony prompts itself for this model; clients send ordinary chat
# messages and must NOT apply a chat template of their own. OpenAI's
# recommended sampling is temperature 1.0, top_p 1.0.
#
# Thinking cannot be turned off for this model, only sized with
# `reasoning_effort`: "low" | "medium" | "high". vLLM has no server flag for a
# default (the model's own default is medium), so gpt_oss_default_effort.py is
# loaded as middleware and fills in "low" when a request leaves the field out.
# A request that sets its own value still wins. Change the server default with
#   GPT_OSS_REASONING_EFFORT=medium ./Model_hosting/deploy_gpt_oss_20b.sh 0
# The reasoning parser moves the thinking into `reasoning_content`, so `content`
# holds only the final answer. Give max_tokens room for both.
#
#   ./Model_hosting/deploy_gpt_oss_20b.sh              # all four cards, port 7472
#   ./Model_hosting/deploy_gpt_oss_20b.sh 0            # a single card
#   ./Model_hosting/deploy_gpt_oss_20b.sh 0,1 7473     # two cards, another port
# Deploy on empty cards only.
set -euo pipefail

CUDA_DEVICES="${1:-0,1,2,3}"
PORT="${2:-7472}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
echo "gpt-oss-20b: $N_GPUS replica(s), one per GPU, on $CUDA_DEVICES, port $PORT"

# One compile cache per GPU model; see deploy_qwen35_35b_fp8.sh for why.
GPU_TAG=$(nvidia-smi -i "${CUDA_DEVICES%%,*}" --query-gpu=name --format=csv,noheader | tr -c 'A-Za-z0-9\n' '_')
export VLLM_CACHE_ROOT="/nas-ssd2/tianyin4/cache/vllm_by_gpu/$GPU_TAG"
echo "vLLM cache: $VLLM_CACHE_ROOT"

# Busy-card check and NCCL settings; see deploy_qwen35_35b_fp8.sh for why.
nvidia-smi -i "$CUDA_DEVICES" --query-gpu=index,memory.used,memory.total --format=csv
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Environment: the vllm-host uv venv (vLLM 0.29, torch 2.13, NCCL 2.29 for
# CUDA 13), on servers whose driver supports CUDA 13. Do NOT preload the CUDA 12
# NCCL copy that the Ada scripts use: it is 2.27.5 and this torch needs symbols
# it lacks (ncclCommResume), so `import torch` fails with it.

# Let vLLM import the middleware that sits next to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export GPT_OSS_REASONING_EFFORT="${GPT_OSS_REASONING_EFFORT:-low}"
echo "default reasoning_effort: $GPT_OSS_REASONING_EFFORT"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" vllm serve "openai/gpt-oss-20b" \
  --host localhost --port "$PORT" \
  --download-dir /nas-ssd2/tianyin4/cache/pretrained_models \
  --max-model-len 32768 \
  --data-parallel-size "$N_GPUS" \
  --enable-prefix-caching \
  --reasoning-parser openai_gptoss \
  --enable-auto-tool-choice --tool-call-parser openai \
  --middleware gpt_oss_default_effort.DefaultReasoningEffort \
  --gpu-memory-utilization 0.9

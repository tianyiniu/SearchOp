#!/usr/bin/env bash
# openai/gpt-oss-20b on a rented GPU server (e.g. Vast.ai). The model, port (7472 by default), window
# (32,768 tokens), reasoning parser and memory settings are those of deploy_gpt_oss_20b.sh, which
# serves this model on our own servers and is left as it is; only the placement differs:
#   - the weights (about 13 GB) go to /workspace/models (Vast.ai's large volume) and vLLM's compile
#     cache to /workspace/vllm_cache when /workspace exists, else to the Hugging Face and vLLM
#     defaults (~/.cache);
#   - one GPU by default (GPU 0); a list such as 0,1 starts one copy per GPU;
#   - none of our servers' NAS paths or NCCL workaround.
#
#     bash Model_hosting/deploy_gpt_oss_20b_rental.sh            # GPU 0, port 7472
#     bash Model_hosting/deploy_gpt_oss_20b_rental.sh 0,1        # GPUs 0 and 1
#     bash Model_hosting/deploy_gpt_oss_20b_rental.sh 0 7474     # another port
#
# The pipeline and run_baselines.py expect port 7472. The server is ready when
# http://localhost:7472/v1/models answers.
set -euo pipefail

CUDA_DEVICES="${1:-0}"
PORT="${2:-7472}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
echo "gpt-oss-20b: $N_GPUS copy/copies, one per GPU, on $CUDA_DEVICES, port $PORT"
if [[ -d /workspace ]]; then
    export HF_HUB_CACHE=/workspace/models VLLM_CACHE_ROOT=/workspace/vllm_cache
    mkdir -p "$HF_HUB_CACHE" "$VLLM_CACHE_ROOT"
fi
echo "weights: ${HF_HUB_CACHE:-the Hugging Face cache}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"     # as deploy_gpt_oss_20b.sh
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" exec vllm serve "openai/gpt-oss-20b" \
  --host localhost --port "$PORT" \
  --max-model-len 32768 \
  --data-parallel-size "$N_GPUS" \
  --enable-prefix-caching \
  --reasoning-parser openai_gptoss \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --async-scheduling

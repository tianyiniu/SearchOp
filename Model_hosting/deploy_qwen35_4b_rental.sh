#!/usr/bin/env bash
# Qwen/Qwen3.5-4B on a rented GPU server (e.g. Vast.ai). The model, port (7473), window (32,768
# tokens), parsers, chat options and memory settings are those of deploy_qwen35_4b.sh, which serves
# this model on our own servers and is left as it is; only the placement differs:
#   - the weights (about 9 GB) go to /workspace/models (Vast.ai's large volume) when /workspace
#     exists, else to the Hugging Face cache (~/.cache/huggingface, or $HF_HOME);
#   - one GPU by default (GPU 0); a list such as 0,1 starts one copy per GPU;
#   - none of our servers' NAS paths or NCCL workaround.
#
#     bash Model_hosting/deploy_qwen35_4b_rental.sh          # GPU 0
#     bash Model_hosting/deploy_qwen35_4b_rental.sh 0,1      # GPUs 0 and 1
#
# The server is ready when http://localhost:7473/v1/models answers.
set -euo pipefail

CUDA_DEVICES="${1:-0}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
echo "Qwen/Qwen3.5-4B: $N_GPUS copy/copies, one per GPU, on $CUDA_DEVICES, port 7473"
if [[ -d /workspace ]]; then
    export HF_HUB_CACHE=/workspace/models
    mkdir -p "$HF_HUB_CACHE"
fi
echo "weights: ${HF_HUB_CACHE:-the Hugging Face cache}"

FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" exec vllm serve "Qwen/Qwen3.5-4B" \
  --trust-remote-code --host localhost --port 7473 \
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

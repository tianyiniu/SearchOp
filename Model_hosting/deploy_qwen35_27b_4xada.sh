#!/usr/bin/env bash
# Qwen3.5-27B (bf16) on 4 x RTX 6000 Ada (48 GB each, no NVLink).
#
# The bf16 checkpoint is ~54 GB, so it cannot sit on one 48 GB card. Two
# tensor-parallel pairs, run as two data-parallel replicas, is the split that
# crosses PCIe the least while still scheduling two independent batches.
#
#   per card   ~27 GB weights, ~14-15 GB KV cache at 0.9 utilization
#   per replica ~230k tokens of fp8 KV -> 50-75 sequences in flight at the
#              2-5k tokens a debate call uses; ~100-150 across the box
#
# vLLM prints the exact "GPU KV cache size" and "maximum concurrency" at
# startup; set --max-num-seqs (per replica) and the clients' --workers (total)
# from those rather than from the estimates above.
#
# Thinking is OFF by server default (--default-chat-template-kwargs), the same
# switch the earlier deployment used; the debate scripts also ask for it on
# every request. A request can still turn it on for a probe with
#   extra_body={"chat_template_kwargs": {"enable_thinking": True}}
# and --reasoning-parser then splits the trace out of the visible content.
#
# Tool calling stays enabled for datasets that need it (FRAMES-style search).
# The parser name is the one the Qwen3.5 model card lists for vLLM; if the
# server rejects it, `vllm serve --help` prints the accepted parser names.
#
#   ./Model_hosting/deploy_qwen35_27b_4xada.sh            # all four cards
#   ./Model_hosting/deploy_qwen35_27b_4xada.sh 0,1        # one pair only
set -euo pipefail

CUDA_DEVICES="${1:-0,1,2,3}"
N_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
TP=2
if (( N_GPUS % TP != 0 )); then
  echo "need a multiple of $TP GPUs for tensor-parallel pairs; got $N_GPUS ($CUDA_DEVICES)" >&2
  exit 1
fi
DP=$(( N_GPUS / TP ))
echo "Qwen3.5-27B bf16: $DP replica(s) x TP $TP on GPUs $CUDA_DEVICES"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" FLASHINFER_DISABLE_VERSION_CHECK=1 vllm serve "Qwen/Qwen3.5-27B" \
  --trust-remote-code \
  --host localhost \
  --port 7472 \
  --download-dir /nas-ssd2/tianyin4/cache/pretrained_models \
  --max-model-len 32768 \
  --data-parallel-size "$DP" \
  --tensor-parallel-size "$TP" \
  --kv-cache-dtype fp8 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.9

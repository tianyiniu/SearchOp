#!/usr/bin/env bash
set -euo pipefail

MODEL_NICKNAME=""
MAX_MODEL_LEN="32768"  # default

# Parse named command-line arguments
while getopts "m:p:d:l:" opt; do
  case $opt in
    m) MODEL_NICKNAME="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    d) CUDA_DEVICES="$OPTARG" ;;
    l) MAX_MODEL_LEN="$OPTARG" ;;
    *)
      echo "Usage: $0 -m <model_nickname> -p <port> -d <cuda_devices> [-l <max_model_len>]"
      exit 1
      ;;
  esac
done

# Check for required args
if [[ -z "$MODEL_NICKNAME" || -z "$PORT" || -z "$CUDA_DEVICES" ]]; then
  echo "Error: Missing required arguments."
  echo "Usage: $0 -m <model_nickname> -p <port> -d <cuda_devices> [-l <max_model_len>]"
  exit 1
fi

# Derive tensor parallel size from CUDA_DEVICES (e.g., "2,3,4" -> 3)
IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"
TP_SIZE="${#GPU_IDS[@]}"
if [[ "$TP_SIZE" -lt 1 ]]; then
  echo "Error: Invalid -d value '$CUDA_DEVICES'. Expected something like '0' or '2,3,4'."
  exit 1
fi

# Map nicknames to model paths
EXTRA_ARGS=""
case "$MODEL_NICKNAME" in
  Llama8)
    MODEL="meta-llama/Llama-3.1-8B-Instruct"
    ;;
  Qwen25-VL-7)
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    ;;
  Qwen25-3)
    MODEL="Qwen/Qwen2.5-3B-Instruct"
    ;;
  Qwen25-7)
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    ;;
  Qwen25-14)
    MODEL="Qwen/Qwen2.5-14B-Instruct"
    ;;
  Qwen25-32)
    MODEL="Qwen/Qwen2.5-32B-Instruct"
    ;;
  Qwen25-72)
    MODEL="Qwen/Qwen2.5-72B-Instruct"
    ;;
  Qwen3-8)
    MODEL="Qwen/Qwen3-8B"
    ;;
  Qwen3-14)
    MODEL="Qwen/Qwen3-14B"
    ;;
  *)
    echo "Unknown model nickname: $MODEL_NICKNAME"
    exit 1
    ;;
esac

if [[ "$TP_SIZE" -gt 1 ]]; then
  EXTRA_ARGS+=" --tensor-parallel-size $TP_SIZE"
fi

# Set max model length
EXTRA_ARGS+=" --max-model-len $MAX_MODEL_LEN"

# Launch the server
echo "Launching $MODEL on port $PORT with CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
echo "Tensor parallel size: $TP_SIZE"
echo "Max model length: $MAX_MODEL_LEN"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" VLLM_USE_V1=1 vllm serve "$MODEL" \
  --trust-remote-code \
  --host localhost \
  --port "$PORT" \
  --download-dir /nas-ssd2/tianyin4/cache/pretrained_models \
  $EXTRA_ARGS

#!/usr/bin/env bash
CUDA_DEVICES="${CUDA_DEVICES:-6,7}"
TP_SIZE="${TP_SIZE:-2}"
MODEL_DIR=/nas-ssd2/tianyin4/cache/pretrained_models

mkdir -p /nas-ssd2/tianyin4/cache/vllm "$MODEL_DIR"

docker run --rm -it --name qwen35 \
  --gpus "\"device=${CUDA_DEVICES}\"" \
  --ipc=host \
  --shm-size 16g \
  -p 127.0.0.1:7472:7472 \
  -v "$MODEL_DIR:$MODEL_DIR" \
  -v /nas-ssd2/tianyin4/cache/vllm:/root/.cache/vllm \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 7472 \
  --download-dir "$MODEL_DIR" \
  --max-model-len 32768 \
  --tensor-parallel-size "$TP_SIZE" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
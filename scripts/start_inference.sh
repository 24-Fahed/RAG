#!/bin/bash
# 启动 Inference Worker (GPU Server)
# 用法: bash scripts/start_inference.sh [local|staging|production]

MODE=$1
cd "$(dirname "$0")/.."
export NO_PROXY=localhost,127.0.0.1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "Starting Inference Worker [mode: $MODE]..."
nohup python -m inference.main --mode $MODE > logs/inference.log 2>&1 &
echo "PID: $!"
echo "Config: inference/config/${MODE}.yaml"
echo "Log:    logs/inference.log"
echo "Watch:  tail -f logs/inference.log"

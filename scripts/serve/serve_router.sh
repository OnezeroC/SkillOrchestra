#!/bin/bash
# =============================================================================
# Serve Qwen2.5-3B-Instruct (router model) locally from ~/chenyilin/models/
# =============================================================================
# Usage:
#   ./scripts/serve/serve_router.sh           # start
#   ./scripts/serve/serve_router.sh --stop    # stop
#   ./scripts/serve/serve_router.sh --status  # check
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate sglang_env

MODEL_PATH="$HOME/chenyilin/models/Qwen2.5-3B-Instruct"
PORT=30008
GPU=0
LOG_DIR="$SO_ROOT/logs/sglang"
LOG_FILE="$LOG_DIR/router_qwen3b.log"

mkdir -p "$LOG_DIR"

case "${1:-}" in
    --stop)
        echo "Stopping router (port $PORT)..."
        pkill -9 -f "port $PORT" 2>/dev/null || true
        echo "Done."
        ;;
    --status)
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Router running on port $PORT"
        else
            echo "Router NOT running on port $PORT"
        fi
        ;;
    *)
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Router already running on port $PORT"
            exit 0
        fi

        if [ ! -d "$MODEL_PATH" ]; then
            echo "ERROR: Model not found at $MODEL_PATH"
            echo "Download first: huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $MODEL_PATH"
            exit 1
        fi

        echo "Starting router: Qwen2.5-3B-Instruct on port $PORT (GPU $GPU)"
        CUDA_VISIBLE_DEVICES=$GPU nohup python -m sglang.launch_server \
            --model-path "$MODEL_PATH" \
            --port $PORT \
            --host 0.0.0.0 \
            --tp 1 \
            --mem-fraction-static 0.30 \
            --trust-remote-code \
            --disable-cuda-graph \
            > "$LOG_FILE" 2>&1 &

        echo "PID: $!"
        echo "Log: $LOG_FILE"
        echo "Waiting for health check..."
        for i in $(seq 1 60); do
            if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
                echo "Router ready! (${i}s)"
                exit 0
            fi
            sleep 2
        done
        echo "WARNING: health check timed out, check $LOG_FILE"
        ;;
esac

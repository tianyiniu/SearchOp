#!/usr/bin/env bash

# GPU indices to track
GPU_IDS="0,1"

# Thresholds to determine if a GPU is "free"
MAX_MEM_MB=2000   # Megabytes used
MAX_UTIL_PCT=5   # Core utilization percentage
INTERVAL=30      # Seconds between checks

# Check if a command was passed as arguments
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command to run when free>"
    echo "Example: $0 python train.py --batch-size 32"
    exit 1
fi

COMMAND=("$@")

echo "Monitoring GPUs ${GPU_IDS} every ${INTERVAL}s..."
echo "Pending command: ${COMMAND[*]}"

while true; do
    # Query memory.used and utilization.gpu for specified GPUs
    stats=$(nvidia-smi --id="${GPU_IDS}" \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits)

    all_free=true

    while IFS=',' read -r idx mem util; do
        idx=$(echo "$idx" | xargs)
        mem=$(echo "$mem" | xargs)
        util=$(echo "$util" | xargs)

        if [ "$mem" -gt "$MAX_MEM_MB" ] || [ "$util" -gt "$MAX_UTIL_PCT" ]; then
            all_free=false
        fi
    done <<< "$stats"

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    if [ "$all_free" = true ]; then
        echo -e "\n[$timestamp] GPUs ${GPU_IDS} are FREE."
        echo "Executing: ${COMMAND[*]}"
        echo "----------------------------------------"
        
        # Replace shell process with target command (or run directly)
        exec "${COMMAND[@]}"
    else
        echo "[$timestamp] In use. Waiting ${INTERVAL}s..."
    fi

    sleep "$INTERVAL"
done
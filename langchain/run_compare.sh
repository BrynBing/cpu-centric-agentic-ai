#!/bin/bash

set -e

ROOT=/home/konnext/cpu-centric-agentic-ai/langchain
RESULTS="$ROOT/results"

cd "$ROOT"

mkdir -p "$RESULTS"

echo "========================================"
echo "Static N experiments"
echo "========================================"

for N in 8 16 32 64
do
    echo "Running Static N=$N"

    python static_scheduler.py \
        --n "$N" \
        --jobs 128 \
        --output "$RESULTS/static_n${N}.csv"

    echo "Static N=$N finished"
    sleep 10
done


echo "========================================"
echo "Dynamic N experiment"
echo "========================================"

python dynamic_scheduler.py \
    --initial-n 64 \
    --min-n 4 \
    --max-n 64 \
    --step 4 \
    --low-cpu 65 \
    --high-cpu 85

sleep 10


echo "========================================"
echo "Analysis"
echo "========================================"

python analyze_results.py

echo "All experiments completed."
#!/usr/bin/env bash

# run_pipeline.sh – orchestrates the 4 modules of ATSME v5.2
# Assumes all Python scripts are installed with required dependencies.
# Zero‑IPC: each module reads/writes JSON/MP4 in the intermediate directory.

set -euo pipefail

# -------------------------------------------------------------------
# 1. STATIC PIPELINE (Module 1)
# -------------------------------------------------------------------
echo "[Stage 1] Running static pipeline..."
python "$(dirname "$0")/module1_static.py" \
    --video "$1" \
    --output-dir "$(dirname "$0")/intermediate"

# -------------------------------------------------------------------
# 2. DYNAMIC PRODUCER (Module 2A)
# -------------------------------------------------------------------
echo "[Stage 2] Running dynamic producer..."
python "$(dirname "$0")/module2a_producer.py" \
    --video "$1" \
    --shots "$(dirname "$0")/intermediate/shots.json" \
    --output-dir "$(dirname "$0")/intermediate"

# -------------------------------------------------------------------
# 3. DYNAMIC CONSUMER (Module 2B)
# -------------------------------------------------------------------
echo "[Stage 3] Running dynamic consumer (action recognition)..."
python "$(dirname "$0")/module2b_consumer.py" \
    --tracklets "$(dirname "$0")/intermediate/tracklets.json" \
    --video "$1" \
    --output-dir "$(dirname "$0")/intermediate"

# -------------------------------------------------------------------
# 4. GRAPH BUILDER (Module 3)
# -------------------------------------------------------------------
echo "[Stage 4] Building knowledge graph..."
python "$(dirname "$0")/module3_graph_builder.py" \
    --input-dir "$(dirname "$0")/intermediate" \
    --output-dir "$(dirname "$0")/output"

echo "Pipeline completed successfully."

#!/bin/bash
# ============================================================
# DepthMaster perspective evaluation.
#
# Runs the perspective-image benchmarks defined in
#   configs/eval/all_benchmarks.json
# (NYUv2, KITTI, ETH3D, iBims-1, GSO, Sintel, DDAD, DIODE, Spring, HAMMER).
#
# Usage:
#   bash eval_perspective.sh <pretrained.pt> [output.json]
# ============================================================
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pretrained.pt> [output.json]"
    exit 1
fi

PRETRAINED=$1
OUTPUT=${2:-output/eval_perspective.json}

# Make the local package importable.
export PYTHONPATH=$(pwd):${PYTHONPATH}

mkdir -p "$(dirname ${OUTPUT})"

python depthmaster/scripts/eval_baseline.py \
    --baseline baselines/depthmaster.py \
    --config configs/eval/all_benchmarks.json \
    --output ${OUTPUT} \
    --pretrained ${PRETRAINED} \
    --resolution_level 9 \
    --fp16

#!/bin/bash
# ============================================================
# DepthMaster panoramic evaluation.
#
# Renders the input panorama as a cubemap, runs the DepthMaster (perspective)
# model on each face, then re-projects back to ERP for evaluation against the
# benchmarks defined in `eval_panorama/configs/eval_panorama.json`
# (Stanford2D3DS, Matterport3D, PanoSUNCG).
#
# Usage:
#   bash eval_panorama.sh <pretrained.pt> [output_dir]
# ============================================================
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pretrained.pt> [output_dir]"
    exit 1
fi

PRETRAINED=$1
OUTPUT_DIR=${2:-output/eval_panorama}

# Make the local package importable.
export PYTHONPATH=$(pwd):$(pwd)/eval_panorama:${PYTHONPATH}

mkdir -p ${OUTPUT_DIR}

python eval_panorama/eval.py \
    --config eval_panorama/configs/eval_panorama.json \
    --pretrained ${PRETRAINED} \
    --cubemap_size 518 \
    --fov_deg 95.0 \
    --output_dir ${OUTPUT_DIR}

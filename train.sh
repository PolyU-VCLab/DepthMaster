#!/bin/bash
# ============================================================
# DepthMaster training launcher (single-/multi-node DDP).
#
# Usage:
#   bash train.sh <node_rank> <nnodes> <master_addr> [extra_hydra_overrides...]
#
# Examples:
#   # single-node, 8 GPUs
#   bash train.sh 0 1 127.0.0.1
#
#   # 2-node cluster, master at 192.168.1.1
#   bash train.sh 0 2 192.168.1.1     # on node 0
#   bash train.sh 1 2 192.168.1.1     # on node 1
#
#   # override training steps via Hydra
#   bash train.sh 0 1 127.0.0.1 trainer.max_steps=200000
# ============================================================
set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <node_rank> <nnodes> <master_addr> [extra_hydra_overrides...]"
    exit 1
fi

# -------------------- Distributed args --------------------
export NODE_RANK=$1
export NNODES=$2
export MASTER_ADDR=$3
export MASTER_PORT=${MASTER_PORT:-23425}
shift 3
EXTRA_ARGS="$@"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))

# -------------------- Runtime envs --------------------
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Make the local package importable (used by hydra _target_ strings).
export PYTHONPATH=$(pwd):$(pwd)/depthmaster:${PYTHONPATH}

echo "=============================================="
echo "  DepthMaster training"
echo "=============================================="
echo "  Node Rank:       $NODE_RANK"
echo "  Number of Nodes: $NNODES"
echo "  GPUs per Node:   $GPUS_PER_NODE"
echo "  World Size:      $WORLD_SIZE"
echo "  Master Address:  $MASTER_ADDR:$MASTER_PORT"
echo "  Extra Args:      $EXTRA_ARGS"
echo "=============================================="

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    training/launch.py \
    --config-name=train.yaml \
    train=depthmaster \
    wrapper=depthmaster \
    data=depthmaster \
    task_name=depthmaster \
    trainer.num_nodes=$NNODES \
    $EXTRA_ARGS

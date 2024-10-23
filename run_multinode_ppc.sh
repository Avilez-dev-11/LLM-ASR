#!/bin/bash
eval "$(~/scratch/miniconda3ppc/bin/conda shell.bash hook)"
module use /gpfs/u/software/dcs-2024/modulefiles
module use cuda/12.1
conda activate venv_ppc
cd scratch-shared/partial-asr

# Set the following environment variables on the head node:
export HEAD_NODE_NUMBER=246
export NUM_GPUS_PER_NODE=6
export NUM_NODES=1
export OMP_NUM_THREADS=$NUM_NODES

# Only change the rank for the non-head nodes:

export NODE_RANK=164

export HEAD_NODE_IP="127.31.134."$HEAD_NODE_NUMBER

torchrun \
--nproc_per_node=$NUM_GPUS_PER_NODE \
--nnodes=$NUM_NODES \
--node_rank=$NODE_RANK \
--rdzv_id=456 \
--rdzv_endpoint=$HEAD_NODE_IP:6000 \
--rdzv_backend=c10d \
finetune_on_the_fly.py




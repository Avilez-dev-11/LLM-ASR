source activate asr
cd scratch-shared/partial-asr
source venv/bin/activate

export OMP_NUM_THREADS=10

# Set the following environment variables on the head node:
export HEAD_NODE_NUMBER=5
export NUM_GPUS_PER_NODE=8
export NUM_NODES=5


# Only change the rank for the non-head node
 export NODE_RANK=5

export HEAD_NODE_IP="172.31.234."$HEAD_NODE_NUMBER

torchrun \
--nproc_per_node=$NUM_GPUS_PER_NODE \
--nnodes=$NUM_NODES \
--node_rank=$NODE_RANK \
--rdzv_id=456 \
--rdzv_endpoint=$HEAD_NODE_IP:6000 \
--rdzv_backend=c10d \
finetune_on_the_fly.py




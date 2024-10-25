#!/bin/bash -x
#SBATCH --nodes=2
#SBATCH --job-name=ai-multi
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --output=o.out
#SBATCH --error=e.out
#SBATCH --time=06:00:00
#SBATCH --partition=npl-2024
#SBATCH --gres=gpu:4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# srun doesnot inherit cpus-per-task from sbatch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
# so processes know who to talk to
# Allow communication over InfiniBand cells.
# Get IP for hostname.
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export MASTER_ADDR=$head_node_ip
# MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_PORT=7010
export GPUS_PER_NODE=4
export NNODES=$SLURM_JOB_NUM_NODES
export WORLDSIZE=8

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1

# handle timeouts
export NCCL_IB_TIMEOUT=20

source activate asr
# Make sure we are on the right directory
cd scratch-shared/partial-asr

# This loads modules and python packages
source venv/bin/activate

export LOGLEVEL=INFO

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"

# Run the demo
time srun bash -c 'accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --multi_gpu \
    --mixed_precision no \
    --num_processes $WORLDSIZE \
    --dynamo_backend no \
    --num_machines $NNODES  \
    --machine_rank $SLURM_PROCID \
    --rdzv_conf "rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT rdzv_backend=c10d" \
    finetune.py'

#accelerate launch \
#    --main_process_ip $MASTER_ADDR \
#    --main_process_port $MASTER_PORT \
#    --multi_gpu \
#    --mixed_precision no \
#    --num_processes $WORLDSIZE \
#    --dynamo_backend no \
#    --num_machines $NNODES  \
#    --machine_rank $SLURM_PROCID \
#    --rdzv_conf "rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT rdzv_backend=c10d" \
#    finetune.py
# srun torchrun \
# --nnodes $NNODES \
# --nproc_per_node $GPUS_PER_NODE \
# --rdzv_id 456 \
# --rdzv_backend c10d \
# --rdzv_endpoint $head_node_ip:29500 \
# finetune.py

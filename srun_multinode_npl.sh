#!/bin/bash
#SBATCH --job-name=large-npl         # Job name
#SBATCH --partition=npl-2024         # Partition
#SBATCH --nodes=1                   # Number of nodes (matching NUM_NODES)
#SBATCH --ntasks-per-node=8          # Number of tasks (GPUs) per node
#SBATCH --cpus-per-task=8           # Number of CPUs per task
#SBATCH --gres=gpu:8                 # GPUs per node
#SBATCH --time=06:00:00              # Max runtime (HH:MM:SS)
#SBATCH --mail-type=begin            # Send email when job begins
#SBATCH --mail-type=end              # Send email when job ends
#SBATCH --mail-user=cereal@phiota.net   # Replace with your email

# Load the necessary environment and modules
source activate asr
cd scratch-shared/partial-asr
source venv/bin/activate

# Set the environment variables for torchrun distributed setup
export OMP_NUM_THREADS=10
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export NODE_RANK=$SLURM_NODEID

# Print distributed settings for reference
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "NODE_RANK="$NODE_RANK

# Run the distributed training using torchrun
srun torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$WORLD_SIZE \
    --node_rank=$NODE_RANK \
    --rdzv_id=456 \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend=c10d \
    finetune_on_the_fly.py

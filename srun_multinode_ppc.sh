#!/bin/bash
#SBATCH --job-name=leg-large-ppc     # create a short name for your job
#SBATCH --partition=dcs-2024     # appropriate partition; if not specified, slurm will automatically do it for you
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # set this equals to the number of gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2             # number of allocated gpus per node
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=omarlcobas@gmail.com    # change this to your email!

# move into the correct directory and set up the environment to run in
eval "$(~/scratch/miniconda3ppc/bin/conda shell.bash hook)"
module use /gpfs/u/software/dcs-2024/modulefiles
module use cuda/12.1
conda activate venv_ppc
cd scratch-shared/partial-asr

# mkdir -p $HOME/.local/share
# mkdir $HOME/scratch/enroot
# ln -s $(readlink $HOME/scratch)/enroot $HOME/.local/share/enroot
# export ENROOT_CACHE_PATH=/gpfs/u/home/NLUG/NLUGfbbr/scratch-shared/partial-asr/enroot/cache
# export ENROOT_DATA_PATH=/gpfs/u/home/NLUG/NLUGfbbr/scratch-shared/partial-asr/enroot/data
# enroot import 'docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel'

# export your rank 0 information (its address and port)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NODE_RANK=$SLURM_NODEID
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "NODE_RANK="$NODE_RANK

srun torchrun \
--nproc_per_node=$SLURM_NTASKS_PER_NODE \
--nnodes=$WORLD_SIZE \
--node_rank=$NODE_RANK \
--rdzv_id=456 \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
--rdzv_backend=c10d \
finetune_on_the_fly.py

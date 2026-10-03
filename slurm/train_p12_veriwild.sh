#!/bin/bash
#SBATCH --job-name=veriwild-p12-ddp2
#SBATCH --output=logs/veriwild_p12_ddp2_%j.log
#SBATCH --error=logs/veriwild_p12_ddp2_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=debug
#SBATCH --gres=gpu:2

# =============================================================================
# Experiment: ViT-Base/12 (patch12, stride12), ImageNet-pretrained,
# trained on VeRi-Wild for 100 epochs with DistributedDataParallel on 2 GPUs.
# Matched against the other patch-size runs in every hyperparameter except
# patch size itself.
#
# Batch: 128 images per GPU (32 IDs x 4 instances), same as the 1-GPU run.
# SOLVER.IMS_PER_BATCH is the *global* batch and is split evenly across ranks,
# so it is set to PER_GPU_BATCH * NUM_GPUS = 256.
#
# The 1-GPU version of this run is slurm/train_p12_veriwild_1gpu.sh.
#
# Usage:
#   sbatch slurm/train_p12_veriwild.sh
# =============================================================================

set -euo pipefail

# --- Paths ---
PROJECT_DIR="$HOME/TransReID/TransReID"

# --- DDP layout ---
NUM_GPUS=2
PER_GPU_BATCH=128
GLOBAL_BATCH=$((PER_GPU_BATCH * NUM_GPUS))

# --- Modules (match whatever `module avail` shows on this cluster) ---
module purge
module load python/3.11
module load rocm/6.2.4

# --- Environment ---
source "/apps/software/anaconda3/etc/profile.d/conda.sh"
conda activate TransReID
cd "$PROJECT_DIR"
mkdir -p logs

# Each rank spawns NUM_WORKERS dataloader workers; keep intra-op threads from oversubscribing the CPUs.
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NUM_GPUS))

# --- Print job info ---
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Run:     veriwild_p12_ddp2"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Batch:   ${PER_GPU_BATCH}/GPU x ${NUM_GPUS} GPUs = ${GLOBAL_BATCH} global"
python -c "import torch; [print('GPU {}:'.format(i), torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
echo "============================================"

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train.py \
    --config_file configs/VeriWild/vit_base_p12.yml \
    MODEL.DIST_TRAIN True \
    MODEL.DEVICE_ID "('0,1')" \
    SOLVER.IMS_PER_BATCH $GLOBAL_BATCH \
    SOLVER.MAX_EPOCHS 100 \
    SOLVER.RESUME True \
    OUTPUT_DIR "../logs/veriwild_vit_base_p12_ddp2" \
    SOLVER.BASE_LR 0.032

echo "Done at $(date)"
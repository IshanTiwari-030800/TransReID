#!/bin/bash
#SBATCH --job-name=veri-p8-patch
#SBATCH --output=logs/veri_p8_%j.log
#SBATCH --error=logs/veri_p8_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=debug
#SBATCH --gres=gpu:1

# =============================================================================
# Experiment: ViT-Base/8 (patch8, stride8), ImageNet-pretrained,
# trained on VeRi-776 for 100 epochs. Matched against train_p16_control.sh in
# every hyperparameter except patch size itself.
#
# Runs on the physical GPU matching the slot Slurm assigned (see PIN_GPU below).
#
# Usage:
#   sbatch slurm/train_p8_patch.sh                          # GPU Slurm assigns
# =============================================================================

set -euo pipefail

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# --- Paths ---
PROJECT_DIR="$HOME/TransReID/TransReID"

# --- GPU pinning ---
# gres.conf on this node is broken (it lists renderD128-135, but only renderD128 is a real GPU), so Slurm
# doesn't restrict which cards a job can see. Everyone else's jobs run on the physical GPU whose amd-smi
# index equals their Slurm slot (SLURM_JOB_GPUS), so do the same: point ROCm at that card by UUID. Running
# anywhere else leaves Slurm handing our card to other jobs, which double-books it. PIN_GPU can override
# this, but only do so if you know that card is unbooked.
GPU_UUIDS=(GPU-eb46644a8c5d4e1f GPU-d15214a684825ef4 GPU-671a6b2f23cd8ce6 GPU-6796b15664f9f25f
           GPU-17b7be1aa9004ca8 GPU-a5ff767c22fb3096 GPU-428ca7e1eef4565b GPU-9a4d4be11adce22a)
PIN_GPU="${PIN_GPU:-$SLURM_JOB_GPUS}"

# --- Modules (match whatever `module avail` shows on this cluster) ---
module purge
module load python/3.11
module load rocm/6.2.4

# --- Environment ---
source "/apps/software/anaconda3/etc/profile.d/conda.sh"
conda activate TransReID
cd "$PROJECT_DIR"
mkdir -p logs

# After module/conda setup, so nothing can reset it. Slurm sets CUDA_VISIBLE_DEVICES/GPU_DEVICE_ORDINAL
# to 0, which is still correct: the pinned card is the only device ROCm exposes, as device 0.
export ROCR_VISIBLE_DEVICES="${GPU_UUIDS[$PIN_GPU]}"

# --- Print job info ---
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Run:     veri_p8"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "GPU:     physical GPU $PIN_GPU ($ROCR_VISIBLE_DEVICES); Slurm's own index was ${SLURM_JOB_GPUS:-?}"
python -c "import torch; n = torch.cuda.device_count(); assert n == 1, f'expected 1 visible GPU, got {n}'; print('Device: ', torch.cuda.get_device_name(0))"
echo "============================================"

python train.py \
    --config_file configs/VeRi/vit_base_p8.yml \
    DATASETS.ROOT_DIR "/mnt/data/ishant/datasets/VeRI-776/" \
    MODEL.PRETRAIN_PATH "/mnt/data/ishant/pretrained_weights/imagenet_vit8.pth" \
    MODEL.GRAD_CHECKPOINT True \
    SOLVER.MAX_EPOCHS 100 \
    MODEL.DEVICE_ID "('0')" \
    SOLVER.RESUME True

echo "Done at $(date)"
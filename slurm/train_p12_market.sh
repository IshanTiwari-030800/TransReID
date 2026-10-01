#!/bin/bash
#SBATCH --job-name=market-p12-patch
#SBATCH --output=logs/market_p12_%j.log
#SBATCH --error=logs/market_p12_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=debug
#SBATCH --gres=gpu:1

# =============================================================================
# Experiment: ViT-Base/12 (patch12, stride12), ImageNet-pretrained,
# trained on Market-1501 for 100 epochs. Matched against train_p16_control.sh in
# every hyperparameter except patch size itself.
#
# Usage:
#   sbatch slurm/train_p12_market.sh
# =============================================================================

set -euo pipefail

# --- Paths ---
PROJECT_DIR="$HOME/TransReID"

# --- Modules (match whatever `module avail` shows on this cluster) ---
module purge
module load python/3.11
module load rocm/6.2.4

# --- Environment ---
source "/apps/software/anaconda3/etc/profile.d/conda.sh"
conda activate TransReID
cd "$PROJECT_DIR"
mkdir -p logs

# --- Print job info ---
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Run:     market_p12"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
echo "============================================"

python train.py \
    --config_file configs/Market/vit_base_p12.yml \
    SOLVER.MAX_EPOCHS 100 \
    MODEL.DEVICE_ID "('0')"

echo "Done at $(date)"

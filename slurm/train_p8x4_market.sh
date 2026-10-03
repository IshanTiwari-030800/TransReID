#!/bin/bash
#SBATCH --job-name=market-p8x4-patch
#SBATCH --output=logs/market_p8x4_%j.log
#SBATCH --error=logs/market_p8x4_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=debug
#SBATCH --gres=gpu:2

# =============================================================================
# Experiment: ViT-Base with non-square 8x4 patches (8 tall x 4 wide, stride 8x4), ImageNet-pretrained,
# trained on Market-1501 for 100 epochs. Matched against train_p16_control.sh in
# every hyperparameter except patch size itself (the config is a copy of the p8 one). On 256x128 Market images
# this gives a square 32x32 token grid. Weights: pretrained_weights/imagenet_vit8x4.pth, made from the ViT-B/16
# checkpoint by tools/resize_patch_weights.py (the same recipe as imagenet_vit8/12.pth).
#
# SEED sets SOLVER.SEED. The default (1234, the config default) is the original run; any other seed writes to
# its own output dir, and so its own wandb run, named with a _seed<N> suffix.
#
# Runs on one physical GPU matching a slot Slurm assigned, skipping GPU 6 (see PIN_GPU below).
#
# Usage:
#   sbatch slurm/train_p8x4_market.sh                          # seed 1234, like the other original runs
#   sbatch --export=ALL,SEED=42 slurm/train_p8x4_market.sh     # -> ../logs/market_vit_base_p8x4_seed42
# =============================================================================

set -euo pipefail

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# --- Paths ---
PROJECT_DIR="$HOME/TransReID/TransReID"

# --- Seed / run name ---
SEED="${SEED:-1234}"
RUN_NAME="market_vit_base_p8x4"
[ "$SEED" = "1234" ] || RUN_NAME="${RUN_NAME}_seed${SEED}"
OUTPUT_DIR="../logs/$RUN_NAME"          # its basename is also the wandb run name

# --- GPU pinning ---
# gres.conf on this node is broken (it lists renderD128-135, but only renderD128 is a real GPU), so Slurm
# doesn't restrict which cards a job can see. Everyone else's jobs run on the physical GPU whose amd-smi
# index equals their Slurm slot (SLURM_JOB_GPUS), so do the same: point ROCm at that card by UUID. Running
# anywhere else leaves Slurm handing our card to other jobs, which double-books it.
#
# Physical GPU 6 hung a rank mid-step in VeRi-Wild jobs 1652, 1665 and 1666, so the job books one spare slot
# (--gres=gpu:2) and trains on the first of its slots that isn't in AVOID_GPUS.
GPU_UUIDS=(GPU-eb46644a8c5d4e1f GPU-d15214a684825ef4 GPU-671a6b2f23cd8ce6 GPU-6796b15664f9f25f
           GPU-17b7be1aa9004ca8 GPU-a5ff767c22fb3096 GPU-428ca7e1eef4565b GPU-9a4d4be11adce22a)
AVOID_GPUS="${AVOID_GPUS:-6}"
if [ -z "${PIN_GPU:-}" ]; then
    for part in ${SLURM_JOB_GPUS//,/ }; do              # expand "a-b" ranges too
        for i in $(seq ${part%-*} ${part#*-}); do
            [[ ",$AVOID_GPUS," == *",$i,"* ]] && continue
            PIN_GPU="$i"; break 2
        done
    done
fi
if [ -z "${PIN_GPU:-}" ]; then
    echo "No usable GPU in Slurm slots ${SLURM_JOB_GPUS:-?} (avoiding $AVOID_GPUS)" >&2
    exit 1
fi

# --- Modules (match whatever `module avail` shows on this cluster) ---
module purge
module load python/3.11
module load rocm/6.2.4

# --- Environment ---
source "/apps/software/anaconda3/etc/profile.d/conda.sh"
conda activate TransReID
cd "$PROJECT_DIR"
mkdir -p logs

# After module/conda setup, so nothing can reset it. The pinned card is then the only device ROCm exposes,
# as device 0; set the HIP-level lists to match whatever Slurm put there.
export ROCR_VISIBLE_DEVICES="${GPU_UUIDS[$PIN_GPU]}"
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 GPU_DEVICE_ORDINAL=0

# --- Print job info ---
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Run:     $RUN_NAME (seed $SEED)"
echo "GPU:     physical $PIN_GPU ($ROCR_VISIBLE_DEVICES); Slurm slots ${SLURM_JOB_GPUS:-?}, avoiding $AVOID_GPUS"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
python -c "import torch; n = torch.cuda.device_count(); assert n == 1, f'expected 1 visible GPU, got {n}'; print('Device: ', torch.cuda.get_device_name(0))"
echo "============================================"

python train.py \
    --config_file configs/Market/vit_base_p8x4.yml \
    SOLVER.MAX_EPOCHS 100 \
    MODEL.DEVICE_ID "('0')" \
    SOLVER.RESUME True \
    MODEL.GRAD_CHECKPOINT True \
    SOLVER.SEED "$SEED" \
    OUTPUT_DIR "$OUTPUT_DIR"

echo "Done at $(date)"

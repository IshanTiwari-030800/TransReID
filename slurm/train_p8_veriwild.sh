#!/bin/bash
#SBATCH --job-name=veriwild-p8-ddp2
#SBATCH --output=logs/veriwild_p8_ddp2_%j.log
#SBATCH --error=logs/veriwild_p8_ddp2_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=debug
#SBATCH --gres=gpu:3

# =============================================================================
# Experiment: ViT-Base/8 (patch8, stride8), ImageNet-pretrained,
# trained on VeRi-Wild for 100 epochs with DistributedDataParallel on 2 GPUs.
# Matched against the other patch-size runs in every hyperparameter except
# patch size itself.
#
# Batch: 128 images per GPU (32 IDs x 4 instances), same as the 1-GPU run.
# SOLVER.IMS_PER_BATCH is the *global* batch and is split evenly across ranks,
# so it is set to PER_GPU_BATCH * NUM_GPUS = 256.
#
# Activation checkpointing (MODEL.GRAD_CHECKPOINT) is on: patch8 gives 1024 tokens
# per 256x256 image, too many activations for 128 images/GPU otherwise. It only
# trades compute for memory; the gradients are the same.
#
# The 1-GPU version of this run is slurm/train_p8_veriwild_1gpu.sh.
#
# Runs on the physical GPUs matching the slots Slurm assigned, skipping GPU 6 (see PIN_GPUS below).
#
# Usage:
#   sbatch slurm/train_p8_veriwild.sh                              # GPUs Slurm assigns
# =============================================================================

set -euo pipefail

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# --- Hang diagnostics ---
# RCCL setup/topology goes to the .err log. If a collective times out, the PyTorch flight recorder
# dumps the last collectives each rank issued to logs/rccl_trace_<jobid>_rank<N>, which shows which rank
# fell behind and in which collective. For live Python stacks: kill -USR1 <train.py pid> (see train.py).
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# A rank that hangs mid-step is detected after this long and the job exits, so the next chained job can
# resume from the last epoch checkpoint. Must stay above rank 0's evaluation time (~5 min for p8).
export DIST_TIMEOUT_MIN=30
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$HOME/TransReID/TransReID/logs/rccl_trace_${SLURM_JOB_ID}_rank"

# --- Paths ---
PROJECT_DIR="$HOME/TransReID/TransReID"

# --- DDP layout ---
NUM_GPUS=2
PER_GPU_BATCH=128
GLOBAL_BATCH=$((PER_GPU_BATCH * NUM_GPUS))

# --- GPU pinning ---
# gres.conf on this node is broken (it lists renderD128-135, but only renderD128 is a real GPU), so Slurm
# doesn't restrict which cards a job can see. Everyone else's jobs run on the physical GPU whose amd-smi
# index equals their Slurm slot (SLURM_JOB_GPUS), so do the same: point ROCm at those cards by UUID. Running
# anywhere else leaves Slurm handing our cards to other jobs, which double-books them (that is what
# happened to job 1652). PIN_GPUS can override this, but only do so if you know that card is unbooked.
#
# Physical GPU 6 hung rank 1 mid-step in jobs 1652, 1665 and 1666 (rank 0 on GPUs 3/4 never did), so the job
# books one spare slot (--gres=gpu:3) and trains on NUM_GPUS of its slots that aren't in AVOID_GPUS.
GPU_UUIDS=(GPU-eb46644a8c5d4e1f GPU-d15214a684825ef4 GPU-671a6b2f23cd8ce6 GPU-6796b15664f9f25f
           GPU-17b7be1aa9004ca8 GPU-a5ff767c22fb3096 GPU-428ca7e1eef4565b GPU-9a4d4be11adce22a)
AVOID_GPUS="${AVOID_GPUS:-6}"
if [ -z "${PIN_GPUS:-}" ]; then
    PIN_GPUS=""
    n=0
    for part in ${SLURM_JOB_GPUS//,/ }; do              # expand "a-b" ranges too
        for i in $(seq ${part%-*} ${part#*-}); do
            [[ ",$AVOID_GPUS," == *",$i,"* ]] && continue
            [ "$n" -lt "$NUM_GPUS" ] || break
            PIN_GPUS="${PIN_GPUS:+$PIN_GPUS,}$i"; n=$((n + 1))
        done
    done
fi
IFS=, read -ra PIN_LIST <<< "$PIN_GPUS"
if [ "${#PIN_LIST[@]}" -ne "$NUM_GPUS" ]; then
    echo "PIN_GPUS=$PIN_GPUS (from Slurm slots ${SLURM_JOB_GPUS:-?}, avoiding $AVOID_GPUS) lists ${#PIN_LIST[@]} GPUs, expected $NUM_GPUS" >&2
    exit 1
fi
PIN_UUIDS=""
for i in "${PIN_LIST[@]}"; do PIN_UUIDS="${PIN_UUIDS:+$PIN_UUIDS,}${GPU_UUIDS[$i]}"; done

# --- Modules (match whatever `module avail` shows on this cluster) ---
module purge
module load python/3.11
module load rocm/6.2.4

# --- Environment ---
source "/apps/software/anaconda3/etc/profile.d/conda.sh"
conda activate TransReID
cd "$PROJECT_DIR"
mkdir -p logs

# After module/conda setup, so nothing can reset it. The pinned cards are then the only devices ROCm
# exposes, as devices 0..NUM_GPUS-1; set the HIP-level lists to match whatever Slurm put there.
export ROCR_VISIBLE_DEVICES="$PIN_UUIDS"
VISIBLE=$(seq -s, 0 $((NUM_GPUS - 1)))
export HIP_VISIBLE_DEVICES="$VISIBLE" CUDA_VISIBLE_DEVICES="$VISIBLE" GPU_DEVICE_ORDINAL="$VISIBLE"

# Each rank spawns NUM_WORKERS dataloader workers; keep intra-op threads from oversubscribing the CPUs.
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NUM_GPUS))

# --- Print job info ---
echo "============================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Run:     veriwild_p8_ddp2"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Batch:   ${PER_GPU_BATCH}/GPU x ${NUM_GPUS} GPUs = ${GLOBAL_BATCH} global"
echo "GPUs:    physical $PIN_GPUS ($ROCR_VISIBLE_DEVICES); Slurm slots ${SLURM_JOB_GPUS:-?}, avoiding $AVOID_GPUS"
python -c "import torch; n = torch.cuda.device_count(); assert n == $NUM_GPUS, f'expected $NUM_GPUS visible GPUs, got {n}'; [print('GPU {}:'.format(i), torch.cuda.get_device_name(i)) for i in range(n)]"
echo "============================================"

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train.py \
    --config_file configs/VeriWild/vit_base_p8.yml \
    MODEL.DIST_TRAIN True \
    MODEL.GRAD_CHECKPOINT True \
    MODEL.DEVICE_ID "('0,1')" \
    SOLVER.IMS_PER_BATCH $GLOBAL_BATCH \
    SOLVER.MAX_EPOCHS 100 \
    SOLVER.RESUME True \
    OUTPUT_DIR "../logs/veriwild_vit_base_p8_ddp2" \
    SOLVER.BASE_LR 0.032

echo "Done at $(date)"
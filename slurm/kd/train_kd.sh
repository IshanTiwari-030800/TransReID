#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug

# =============================================================================
# One KD run (train_kd.py) on NUM_GPUS GPUs. Submit it through slurm/kd/submit.sh, which sets the job name,
# log paths, GPU/CPU/memory request and the chain of resume jobs. Can also be run under srun for testing:
#
#   srun --gres=gpu:1 --cpus-per-task=16 --mem=64G --partition=debug \
#        bash slurm/kd/train_kd.sh configs/KD/Market/dino_kd.yml [KEY VALUE ...]
#
# NUM_GPUS (env, default 1): GPUs to train on. Above 1 this runs DDP via torch.distributed.run; the config's
# SOLVER.IMS_PER_BATCH is the global batch, split evenly across GPUs.
# Training resumes from OUTPUT_DIR/checkpoint_latest.pth when it exists (SOLVER.RESUME True).
# =============================================================================

set -euo pipefail

CONFIG="$1"; shift
NUM_GPUS="${NUM_GPUS:-1}"
PROJECT_DIR="$HOME/TransReID/TransReID"

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# The CK grouped-conv weight-gradient solver prints "Workspace ... is not allocated" ~3x per iteration for the
# depthwise conv in the deformable offset network; MIOpen's other solvers handle that conv at the same speed.
export MIOPEN_DEBUG_GROUP_CONV_IMPLICIT_GEMM_HIP_WRW_XDLOPS=0
# A rank that hangs mid-step is detected after this long and the job exits, so the next chained job can
# resume. Must stay above rank 0's evaluation time.
export DIST_TIMEOUT_MIN=30

# --- GPU pinning (same scheme as slurm/train_p8_veriwild.sh) ---
# gres.conf on this node doesn't restrict which cards a job sees; everyone runs on the physical GPU whose index
# equals their Slurm slot, so do the same. Slots in AVOID_GPUS (GPU 6 hung DDP ranks in jobs 1652/1665/1666) are
# used only if the job has no other free slot; submit.sh books a spare slot for DDP jobs to make that unlikely.
GPU_UUIDS=(GPU-eb46644a8c5d4e1f GPU-d15214a684825ef4 GPU-671a6b2f23cd8ce6 GPU-6796b15664f9f25f
           GPU-17b7be1aa9004ca8 GPU-a5ff767c22fb3096 GPU-428ca7e1eef4565b GPU-9a4d4be11adce22a)
AVOID_GPUS="${AVOID_GPUS:-6}"
JOB_GPUS="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}"   # sbatch sets the former, a standalone srun the latter
if [ -z "${PIN_GPUS:-}" ]; then
    preferred=(); avoided=()
    for part in ${JOB_GPUS//,/ }; do                    # expand "a-b" ranges too
        for i in $(seq ${part%-*} ${part#*-}); do
            if [[ ",$AVOID_GPUS," == *",$i,"* ]]; then avoided+=("$i"); else preferred+=("$i"); fi
        done
    done
    slots=("${preferred[@]}" "${avoided[@]}")
    PIN_GPUS=$(IFS=,; echo "${slots[*]:0:$NUM_GPUS}")
    if [ "${#preferred[@]}" -lt "$NUM_GPUS" ]; then
        echo "WARNING: only ${#preferred[@]} slot(s) outside AVOID_GPUS=$AVOID_GPUS; using $PIN_GPUS" >&2
    fi
fi
IFS=, read -ra PIN_LIST <<< "$PIN_GPUS"
if [ "${#PIN_LIST[@]}" -ne "$NUM_GPUS" ]; then
    echo "PIN_GPUS=$PIN_GPUS (Slurm slots ${JOB_GPUS:-?}) lists ${#PIN_LIST[@]} GPUs, expected $NUM_GPUS" >&2
    exit 1
fi
PIN_UUIDS=""
for i in "${PIN_LIST[@]}"; do PIN_UUIDS="${PIN_UUIDS:+$PIN_UUIDS,}${GPU_UUIDS[$i]}"; done

# --- Environment ---
module purge
module load python/3.11
module load rocm/6.2.4
source "/apps/software/anaconda3/etc/profile.d/conda.sh"
conda activate TransReID
cd "$PROJECT_DIR"

export ROCR_VISIBLE_DEVICES="$PIN_UUIDS"
VISIBLE=$(seq -s, 0 $((NUM_GPUS - 1)))
export HIP_VISIBLE_DEVICES="$VISIBLE" CUDA_VISIBLE_DEVICES="$VISIBLE" GPU_DEVICE_ORDINAL="$VISIBLE"
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / NUM_GPUS ))

echo "============================================"
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "Config:  $CONFIG $*"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Commit:  $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!kd_runs' || echo ' (dirty)')"
echo "GPUs:    physical $PIN_GPUS ($ROCR_VISIBLE_DEVICES); Slurm slots ${JOB_GPUS:-?}"
python -c "import torch; n = torch.cuda.device_count(); assert n == $NUM_GPUS, f'expected $NUM_GPUS visible GPUs, got {n}'; [print('GPU {}:'.format(i), torch.cuda.get_device_name(i)) for i in range(n)]"
echo "============================================"

if [ "$NUM_GPUS" -gt 1 ]; then
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train_kd.py \
        --config_file "$CONFIG" MODEL.DIST_TRAIN True MODEL.DEVICE_ID "('$VISIBLE')" SOLVER.RESUME True "$@"
else
    python train_kd.py --config_file "$CONFIG" SOLVER.RESUME True "$@"
fi

echo "Done at $(date)"

#!/bin/bash
# =============================================================================
# Submit one KD run as a chain of 24 h Slurm jobs; each job resumes from the previous one's checkpoint, and a job
# that starts after training already finished exits immediately.
#
#   slurm/kd/submit.sh <market|veri|veriwild> <baseline|soft_kd|dino_kd> [CHAIN_LENGTH]
#
# Config: configs/KD/<Market|VeRi|VeriWild>/<method>.yml
# Output: kd_runs/<market1501|veri776|veriwild>/<method>/ (train_log.txt, metrics.jsonl, config.yaml,
#         slurm_<jobid>.log/.err, and the git-ignored *.pth checkpoints)
#
# Env: SPARE_GPUS  extra Slurm slots to book so the job can skip GPU 6 (default 1 for DDP runs, 0 otherwise)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASET="${1:?dataset: market, veri or veriwild}"
METHOD="${2:?method: baseline, soft_kd or dino_kd}"

# Chain lengths cover the estimated run time (see kd/README.md) with about one job of slack.
case "$DATASET" in
    market)   CFG_DIR=Market;   OUT=market1501; NUM_GPUS=1; CHAIN=1 ;;
    veri)     CFG_DIR=VeRi;     OUT=veri776;    NUM_GPUS=1; CHAIN=2 ;;
    veriwild) CFG_DIR=VeriWild; OUT=veriwild;   NUM_GPUS=2; CHAIN=3 ;;
    *) echo "unknown dataset $DATASET" >&2; exit 1 ;;
esac
CHAIN="${3:-$CHAIN}"
CONFIG="configs/KD/$CFG_DIR/$METHOD.yml"
[ -f "$CONFIG" ] || { echo "no config $CONFIG" >&2; exit 1; }
RUN_DIR="kd_runs/$OUT/$METHOD"
mkdir -p "$RUN_DIR"   # Slurm doesn't create the directory for --output/--error

SPARE_GPUS="${SPARE_GPUS:-$([ "$NUM_GPUS" -gt 1 ] && echo 1 || echo 0)}"
prev=""
for i in $(seq 1 "$CHAIN"); do
    prev=$(sbatch --parsable ${prev:+--dependency=afterany:$prev} \
        --job-name="kd-$DATASET-$METHOD" \
        --gres="gpu:$((NUM_GPUS + SPARE_GPUS))" \
        --cpus-per-task=$((16 * NUM_GPUS)) --mem=$((64 * NUM_GPUS))GB \
        --output="$RUN_DIR/slurm_%j.log" --error="$RUN_DIR/slurm_%j.err" \
        --export=ALL,NUM_GPUS="$NUM_GPUS" \
        slurm/kd/train_kd.sh "$CONFIG")
    echo "kd-$DATASET-$METHOD: job $prev ($i/$CHAIN)"
done

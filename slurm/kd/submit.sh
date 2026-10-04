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
# W&B runs are named <dataset>_<method> (e.g. market1501_dino_kd).
#
# Env: SPARE_GPUS  extra Slurm slots to book so the job can skip GPU 6 (default 1 for DDP runs, 0 otherwise)
#      FINETUNE=1  after the KD chain succeeds, fine-tune its transformer_best.pth without KD
#                  (configs/KD/<dataset>/finetune.yml) into kd_runs/<dataset>/<method>_ft/, W&B run <dataset>_<method>_ft
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
FT_CONFIG="configs/KD/$CFG_DIR/finetune.yml"
if [ "${FINETUNE:-0}" = 1 ]; then
    [ -f "$FT_CONFIG" ] || { echo "FINETUNE=1 but no config $FT_CONFIG" >&2; exit 1; }
fi
sbatch_run() {   # <job name> <run dir> <dependency or ""> <train_kd.sh args...>
    local name=$1 dir=$2 dep=$3; shift 3
    sbatch --parsable ${dep:+--dependency=$dep} \
        --job-name="$name" \
        --gres="gpu:$((NUM_GPUS + SPARE_GPUS))" \
        --cpus-per-task=$((16 * NUM_GPUS)) --mem=$((64 * NUM_GPUS))GB \
        --output="$dir/slurm_%j.log" --error="$dir/slurm_%j.err" \
        --export=ALL,NUM_GPUS="$NUM_GPUS" \
        slurm/kd/train_kd.sh "$@"
}

prev=""
for i in $(seq 1 "$CHAIN"); do
    prev=$(sbatch_run "kd-$DATASET-$METHOD" "$RUN_DIR" "${prev:+afterany:$prev}" \
        "$CONFIG" WANDB.NAME "${OUT}_$METHOD")
    echo "kd-$DATASET-$METHOD: job $prev ($i/$CHAIN)"
done

if [ "${FINETUNE:-0}" = 1 ]; then
    # The last KD job exits 0 once training is done (immediately, if an earlier job finished it).
    FT_DIR="${RUN_DIR}_ft"
    mkdir -p "$FT_DIR"
    ft=$(sbatch_run "ft-$DATASET-$METHOD" "$FT_DIR" "afterok:$prev" \
        "$FT_CONFIG" MODEL.PRETRAIN_PATH "$RUN_DIR/transformer_best.pth" OUTPUT_DIR "$FT_DIR" \
        WANDB.NAME "${OUT}_${METHOD}_ft")
    echo "ft-$DATASET-$METHOD: job $ft (after $prev succeeds)"
fi

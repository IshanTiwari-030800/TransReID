# Knowledge distillation: ViT-B/16 ReID teacher → randomly initialised deformable ViT-B/8

For each dataset, the ViT-Base/16 trained on that dataset is distilled into a **randomly initialised** deformable
ViT-Base with 8×8 patches. Two KD recipes plus a no-KD baseline are implemented and configured independently:

```text
16×16 trained ReID checkpoints (frozen)
        ↓
Market-1501 ──→ Deformable ViT 8×8 + {soft KD | DINO KD | none} ──→ kd_runs/market1501/<method>/
VeRi-776    ──→ Deformable ViT 8×8 + {soft KD | DINO KD | none} ──→ kd_runs/veri776/<method>/
VeRi-Wild   ──→ Deformable ViT 8×8 + {soft KD | DINO KD | none} ──→ kd_runs/veriwild/<method>/
```

| Dataset    | Teacher config                              | Teacher weights (`/mnt/data/ishant/TransReID/logs/…`)     | Teacher mAP / R1 |
|------------|---------------------------------------------|-----------------------------------------------------------|------------------|
| Market-1501| `configs/Market/vit_base_p16_control.yml`   | `market_vit_base_p16_control/transformer_best.pth`        | 87.3 / 94.6      |
| VeRi-776   | `configs/VeRi/vit_base_p16_control.yml`     | `veri_vit_base_p16_control/transformer_best.pth`          | 79.1 / 96.7      |
| VeRi-Wild  | `configs/VeriWild/vit_base_p16_control.yml` | `veriwild_vit_base_p16_control_ddp2/transformer_best.pth` | 79.2 / 91.2      |

Every run re-evaluates its teacher before training (`KD.TEACHER.EVAL_AT_START`) and logs the checkpoint's sha256,
so a wrong or corrupted teacher shows up in the first minute of `train_log.txt` and `metrics.jsonl`.

## Layout

| Path | What |
|---|---|
| `model/backbones/deformable_vit.py` | Deformable ViT-B/8 student (`deformable_vit_base_patch8_TransReID`) |
| `kd/teacher.py` | Rebuilds the teacher from its own yml, loads its weights strictly, freezes it |
| `kd/losses.py` | Soft-KD KL, DINO head, Sinkhorn-Knopp, EMA centering, DINO cross-entropy, KoLeo |
| `kd/methods.py` | The recipes behind one interface: `NoKD`, `SoftKD`, `DINOKD` |
| `processor/kd_processor.py` | Training loop: AMP, DDP, grad clipping, eval, best/latest/resume checkpoints |
| `train_kd.py` | Entry point |
| `configs/KD/<Market,VeRi,VeriWild>/<baseline,soft_kd,dino_kd>.yml` | One config per run |
| `slurm/kd/submit.sh`, `slurm/kd/train_kd.sh` | Slurm submission (chained resume jobs, GPU pinning) |
| `kd_runs/<market1501,veri776,veriwild>/<baseline,soft_kd,dino_kd>/` | Run outputs |

Each run directory holds `train_log.txt`, `metrics.jsonl` (teacher + every student evaluation), `config.yaml`
(the fully merged config, so the run can be reproduced with `--config_file` alone), `slurm_<job>.log/.err`, and the
checkpoints `transformer_best.pth`, `transformer_latest.pth` (student ReID weights, loadable by `test.py`) and
`checkpoint_latest.pth` (full state for resuming). The `.pth` files are git-ignored: each is 350 MB–1 GB, over
GitHub's 100 MB limit. Everything else is meant to be committed.

## Student: deformable ViT-B/8

A plain ViT-Base (12 blocks, 768-d, 12 heads, CLS token, absolute position embeddings, TransReID's BNNeck head)
with 8×8 patches, where every block's self-attention is replaced by DAT-style deformable attention
([Xia et al., CVPR 2022](https://arxiv.org/abs/2201.00520)):

1. a reference grid is laid over the token map, `MODEL.DEFORM.DOWNSAMPLE` (2) times coarser;
2. an offset network (depthwise 5×5 conv → LayerNorm → GELU → 1×1 conv) reads the queries and predicts a 2-D offset per
   reference point, separately for each of `MODEL.DEFORM.GROUPS` (4) channel groups, bounded by tanh to
   ±`MODEL.DEFORM.OFFSET_RANGE` (2) reference cells;
3. features are bilinearly sampled at reference + offset; keys and values are projected from the samples, plus the
   CLS token.

Every token is a query, so the output keeps the token layout and the TransReID head is unchanged. Each query attends
to 1 + 16×8 = 129 keys on Market (256×128 input, 32×16 tokens) and 1 + 16×16 = 257 on VeRi/VeRi-Wild (256×256, 32×32
tokens), instead of 513 / 1025. The last offset conv is zero-initialised, so training starts from uniform-grid sampling
(2×2 average pooling of the tokens) and learns where to look.

Departures from DAT: DAT is a 4-stage hierarchy that alternates local-window and deformable attention, and adds a
relative position bias. This model stays single-stage so it is a drop-in TransReID backbone, and relies on the
absolute position embeddings instead of the bias. `MODEL.DEFORM.BLOCKS` can restrict deformable attention to
some blocks; the rest then use full attention.

Random initialisation: `MODEL.PRETRAIN_CHOICE: 'scratch'`. Nothing is loaded; weights use the ViT
initialisation (truncated normal 0.02).

## Shared training recipe

Within a dataset, all three configs share everything outside the `KD` block: the student, the data pipeline
(TransReID's augmentation, P×K = 32×4 identity sampling), ID + soft-margin triplet losses with BNNeck, and the solver.
The solver is set up for a ViT trained from scratch (the teachers' SGD recipe assumes ImageNet initialisation):
AdamW, weight decay 0.05 (none on biases, norms, position embeddings and the CLS token), linear warmup then cosine,
gradient clipping at 3.0, AMP, seed 1234.

| Dataset   | Epochs | Warmup | Batch (global)      | LR     | Eval every |
|-----------|--------|--------|---------------------|--------|------------|
| Market    | 200    | 10     | 128 (1 GPU)         | 2.5e-4 | 10 epochs  |
| VeRi-776  | 120    | 10     | 128 (1 GPU)         | 2.5e-4 | 10 epochs  |
| VeRi-Wild | 60     | 5      | 256 (2 GPUs × 128)  | 3.5e-4 | 5 epochs   |

The epoch counts give Market (13k images) about 18.6k iterations, VeRi (38k) about 35k and VeRi-Wild (278k) about 57k.
The learning rate scales with the square root of the batch size.

## Recipe 1: soft KD (`KD.METHOD: soft`)

[Hinton et al. 2015](https://arxiv.org/abs/1503.02531). The teacher and student share the training identities, so the
student's ID classifier is distilled directly from the teacher's:

    L = L_ID + L_triplet + KD.SOFT.WEIGHT · T² · KL( softmax(z_t / T) ‖ softmax(z_s / T) )

with T = `KD.SOFT.TEMPERATURE` = 4 and weight 1. Teacher and student see the same augmented image. The log also
tracks student/teacher top-1 agreement and the teacher's batch accuracy.

## Recipe 2: DINO-based KD (`KD.METHOD: dino`)

DINO's self-distillation ([Caron et al. 2021](https://arxiv.org/abs/2104.14294)), with the EMA teacher *backbone*
replaced by the frozen ReID teacher, and the batch-level tools and frozen-teacher distillation setup from DINOv2
([Oquab et al. 2023](https://arxiv.org/abs/2304.07193)):

- **Prototypes (`TEACHER_HEAD: kmeans`).** Before training, the teacher embeds the whole training set (no
  augmentation). Spherical k-means (k-means++ seeding, 30 iterations) on its L2-normalised features gives `OUT_DIM`
  prototypes: 2048 for Market and VeRi (a few per identity), 8192 for VeRi-Wild. They are frozen and shared by both
  branches. In DINOv2's distillation the frozen teacher keeps its trained DINO head. A ReID teacher has no such
  head, and these centroids play that role: the teacher's scores are its features' cosines to the centroids, which
  are meaningful from the first step.
- **Student head.** DINO head on the student's global feature: 3-layer MLP (768→2048→2048→768), L2 normalisation,
  cosine similarity to the prototypes. It learns to project the student's features into the teacher's feature space.
- **Targets.** Teacher prototype scores, balanced with **Sinkhorn-Knopp** (3 iterations, all-reduced across DDP ranks)
  at temperature 0.04 → 0.07 (linear warmup). `KD.DINO.CENTERING: centering` switches to DINO v1's EMA centering.
- **Views.** Each image gets 2 independently augmented views (`INPUT.NUM_VIEWS`). The teacher sees both; the
  student sees `STUDENT_VIEWS` = 1 of them. The loss averages the student view against both teacher views, i.e. the
  same view and the other view. With 1 student view the student does the same work per step as in soft KD and the
  baseline, so the recipes compare at equal student compute. Set `STUDENT_VIEWS: 2` for DINO's symmetric version, at
  twice the student compute.
- **KoLeo** ([Sablayrolles et al. 2019](https://arxiv.org/abs/1806.03198)), weight 0.1, on the student's
  L2-normalised global feature. ReID adaptation: each sample's nearest neighbour is taken among *other* identities
  only (`KOLEO_EXCLUDE_SAME_ID`). In a P×K batch the nearest sample is usually the same identity, and pushing it away
  would fight the triplet loss.

      L = L_ID + L_triplet + KD.DINO.WEIGHT · mean_pairs H(SK(t_j), softmax(s_i / 0.1)) + 0.1 · KoLeo(student feats)

`TEACHER_HEAD: ema` keeps DINO's original scheme instead: learnable prototypes (bottleneck 256), and a teacher head
that is an EMA of the student head (momentum 0.996 → 1), with the prototypes frozen for the first epoch. With a
frozen pretrained teacher this works poorly. The EMA head starts random and only becomes meaningful as fast as the
student does, so the targets stay near-uniform. In a 10-epoch Market test the DINO loss only fell from 8.44 to 8.21
(log K = 8.32), and student/teacher prototype agreement stayed around 1%. Hence the k-means default.

### How collapse is handled

There are two kinds of collapse: every sample mapping to the same prototype or feature, and outputs going uniform.
These are the guards against them:

| Risk | Guard |
|---|---|
| Teacher targets collapse | The teacher backbone and its prototypes are fixed, so its targets can't drift or collapse. |
| All samples assigned to one prototype | Sinkhorn-Knopp makes every prototype receive equal total mass per (global) batch, so this is impossible by construction. A unit test shows SK spreads a deliberately collapsed input over 227 of 1024 prototypes; plain softmax puts all of them on 1. |
| Targets go uniform | Sharp teacher temperature (0.04 → 0.07) vs. student 0.1. |
| Student features shrink to a low-dimensional set | KoLeo spreads the student's features. |
| Unstable start | Gradient clipping and LR warmup. |

Every `LOG_PERIOD` iterations the log and W&B record collapse diagnostics (`collapse/*`):
- per-sample and batch-marginal entropy of the teacher targets and student outputs (log K = 8.32 is uniform; a
  marginal near 0 means everything is on one prototype);
- distinct argmax prototypes per batch;
- `student_feat_std`, the per-dimension std of the normalised student features. About 1/√768 ≈ 0.036 when spread
  out, near 0 when collapsed. A randomly initialised ViT starts near 0 because all images give almost the same CLS
  feature.

## Running

```bash
# One run = a chain of 24 h jobs that resume from checkpoint_latest.pth
slurm/kd/submit.sh market   soft_kd
slurm/kd/submit.sh market   dino_kd
slurm/kd/submit.sh veri     soft_kd
slurm/kd/submit.sh veri     dino_kd
slurm/kd/submit.sh veriwild soft_kd      # 2-GPU DDP
slurm/kd/submit.sh veriwild dino_kd
slurm/kd/submit.sh market   baseline     # optional no-KD reference, same for veri / veriwild

# Interactive / custom (any config key can be overridden on the command line)
srun --gres=gpu:1 --cpus-per-task=16 --mem=64G --partition=debug \
     bash slurm/kd/train_kd.sh configs/KD/Market/dino_kd.yml KD.DINO.KOLEO_WEIGHT 0.0

# Evaluate a trained student
python test.py --config_file configs/KD/Market/soft_kd.yml TEST.WEIGHT kd_runs/market1501/soft_kd/transformer_best.pth
```

## Reproducibility

- `config.yaml` in each run directory is the complete merged config. The log records the git commit
  (flagging uncommitted changes) and the teacher checkpoint's sha256.
- Seeds are fixed (`SOLVER.SEED`, offset by rank under DDP so ranks draw different augmentations). Resuming restores
  the model, KD state (EMA head, centering), optimizer, AMP scaler and all RNG states, so an interrupted run continues
  the same way.
- GPU training is not bitwise deterministic: `grid_sample`'s backward pass and some MIOpen kernels use atomics.
  Expect run-to-run differences in the last digit of mAP.

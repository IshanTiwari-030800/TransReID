from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
# Recompute transformer-block activations in backward to save memory (same math, ~30% slower)
_C.MODEL.GRAD_CHECKPOINT = False
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# Deformable attention (only read by the deformable_* TRANSFORMER_TYPEs, see model/backbones/deformable_vit.py)
_C.MODEL.DEFORM = CN()
# Indices of the blocks whose attention is deformable; empty means every block
_C.MODEL.DEFORM.BLOCKS = []
# Offset groups: each group of channels predicts and samples its own set of offsets
_C.MODEL.DEFORM.GROUPS = 4
# Reference grid is the token grid downsampled by this factor, so each query attends to N / RATIO^2 keys (+CLS)
_C.MODEL.DEFORM.DOWNSAMPLE = 2
# Kernel size of the depthwise conv in the offset network
_C.MODEL.DEFORM.KERNEL = 5
# Maximum offset, in units of reference-grid spacing
_C.MODEL.DEFORM.OFFSET_RANGE = 2.0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# Independently augmented views of each training image (DINO-style KD uses 2; everything else uses 1)
_C.INPUT.NUM_VIEWS = 1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# Max global grad norm (0 disables clipping). Used by train_kd.py only.
_C.SOLVER.CLIP_GRAD = 0.0
# Debugging only: stop each epoch after this many iterations (0 = full epoch). Used by train_kd.py only.
_C.SOLVER.MAX_ITERS_PER_EPOCH = 0

# If True, continue from OUTPUT_DIR/checkpoint_latest.pth when it exists (else start fresh).
# Safe to leave on in sbatch scripts: resubmitting the same script resumes a crashed run.
_C.SOLVER.RESUME = False

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Weights & Biases
# ---------------------------------------------------------------------------- #
_C.WANDB = CN()
_C.WANDB.PROJECT = "transreid"
# Set False to train without logging to Weights & Biases (e.g. smoke tests)
_C.WANDB.ENABLED = True
# Run name; empty uses the basename of OUTPUT_DIR
_C.WANDB.NAME = ''

# ---------------------------------------------------------------------------- #
# Knowledge distillation (train_kd.py). See kd/README.md.
# ---------------------------------------------------------------------------- #
_C.KD = CN()
# 'none' (student trained on ID + triplet only), 'soft' or 'dino'
_C.KD.METHOD = 'none'

_C.KD.TEACHER = CN()
# Config the teacher was trained with (architecture and input size are read from it) and its trained weights
_C.KD.TEACHER.CONFIG = ''
_C.KD.TEACHER.WEIGHT = ''
# Evaluate the teacher on the val set before training, to confirm the weights loaded correctly
_C.KD.TEACHER.EVAL_AT_START = True

# Soft KD (Hinton et al.): KL between temperature-softened teacher and student ID logits
_C.KD.SOFT = CN()
_C.KD.SOFT.TEMPERATURE = 4.0
_C.KD.SOFT.WEIGHT = 1.0

# DINO-style KD: frozen teacher backbone + EMA projection head gives prototype assignments the student matches
_C.KD.DINO = CN()
_C.KD.DINO.WEIGHT = 1.0
# 'kmeans': frozen prototypes = spherical k-means centroids of the teacher's features on the training set.
# 'ema': DINO's teacher head = EMA of the student head. See kd/methods.py:DINOKD.
_C.KD.DINO.TEACHER_HEAD = 'kmeans'
_C.KD.DINO.KMEANS_ITERS = 30
# Number of prototypes
_C.KD.DINO.OUT_DIM = 4096
_C.KD.DINO.HIDDEN_DIM = 2048
# Head output dim for 'ema' ('kmeans' projects to the teacher's feature dim)
_C.KD.DINO.BOTTLENECK_DIM = 256
_C.KD.DINO.STUDENT_TEMP = 0.1
_C.KD.DINO.TEACHER_TEMP = 0.07
_C.KD.DINO.WARMUP_TEACHER_TEMP = 0.04
_C.KD.DINO.WARMUP_TEACHER_TEMP_EPOCHS = 30
# 'sinkhorn_knopp' (balanced assignments, DINOv2/SwAV) or 'centering' (EMA center, DINO v1)
_C.KD.DINO.CENTERING = 'sinkhorn_knopp'
_C.KD.DINO.SK_ITERS = 3
_C.KD.DINO.CENTER_MOMENTUM = 0.9
# 'ema' only: the teacher head's EMA momentum follows a cosine from this value to 1
_C.KD.DINO.HEAD_MOMENTUM = 0.996
# 'ema' only: keep the prototype layer fixed for this many epochs (DINO's stabilising trick)
_C.KD.DINO.FREEZE_LAST_LAYER_EPOCHS = 1
# How many of the INPUT.NUM_VIEWS views the student sees (the teacher always sees all of them)
_C.KD.DINO.STUDENT_VIEWS = 1
# Also match student view i to teacher view i (DINO skips these, but here teacher != student)
_C.KD.DINO.SAME_VIEW_PAIRS = True
# KoLeo regulariser on the student's L2-normalised global feature (0 disables)
_C.KD.DINO.KOLEO_WEIGHT = 0.1
# Take each sample's nearest neighbour among other identities only, so KoLeo doesn't fight the triplet loss
_C.KD.DINO.KOLEO_EXCLUDE_SAME_ID = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""

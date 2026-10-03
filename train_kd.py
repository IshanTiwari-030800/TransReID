"""Knowledge distillation from a trained TransReID teacher into a randomly initialised student. See kd/README.md.

    python train_kd.py --config_file configs/KD/Market/dino_kd.yml [KEY VALUE ...]
"""
import argparse
import datetime
import faulthandler
import os
import signal
import subprocess

import torch

from config import cfg
from datasets import make_dataloader
from kd import build_kd, build_teacher, make_kd_optimizer
from loss import make_loss
from model import make_model
from processor.kd_processor import do_train_kd
from solver.scheduler_factory import create_scheduler
from train import set_seed
from utils.logger import setup_logger

# `kill -USR1 <pid>` prints every Python thread's stack to stderr.
faulthandler.register(signal.SIGUSR1, all_threads=True)


def check_kd_config(cfg):
    method = cfg.KD.METHOD
    if cfg.MODEL.JPM:
        raise ValueError('train_kd.py expects a single global feature; set MODEL.JPM False')
    if method in ('none', 'soft') and cfg.INPUT.NUM_VIEWS != 1:
        raise ValueError('KD.METHOD {} uses one view; set INPUT.NUM_VIEWS 1'.format(method))
    if method == 'dino':
        s, v = cfg.KD.DINO.STUDENT_VIEWS, cfg.INPUT.NUM_VIEWS
        if not 1 <= s <= v:
            raise ValueError('need 1 <= KD.DINO.STUDENT_VIEWS ({}) <= INPUT.NUM_VIEWS ({})'.format(s, v))
        if v == 1 and not cfg.KD.DINO.SAME_VIEW_PAIRS:
            raise ValueError('with one view, KD.DINO.SAME_VIEW_PAIRS must be True or there are no pairs to match')
    if method != 'none' and not (cfg.KD.TEACHER.CONFIG and cfg.KD.TEACHER.WEIGHT):
        raise ValueError('KD.METHOD {} needs KD.TEACHER.CONFIG and KD.TEACHER.WEIGHT'.format(method))


def git_revision():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=here, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here,
                                        text=True).strip()
        return rev + (' (with uncommitted changes)' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TransReID knowledge distillation")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", "--local-rank", default=int(os.environ.get("LOCAL_RANK", 0)), type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    check_kd_config(cfg)

    set_seed(cfg.SOLVER.SEED + int(os.environ.get("RANK", 0)))

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=True, resume=cfg.SOLVER.RESUME)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    logger.info("Code revision: {}".format(git_revision()))
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))
    if int(os.environ.get("RANK", 0)) == 0:
        # The fully merged config (yml + command-line overrides): rerunning with it reproduces this run.
        with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as f:
            f.write(cfg.dump())

    if cfg.MODEL.DIST_TRAIN:
        timeout_min = int(os.environ.get('DIST_TIMEOUT_MIN', 120))
        torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                             timeout=datetime.timedelta(minutes=timeout_min))

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    student = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    teacher = None
    if cfg.KD.METHOD != 'none':
        teacher, _ = build_teacher(cfg, num_classes, camera_num, view_num)

    iters_per_epoch = min(cfg.SOLVER.MAX_ITERS_PER_EPOCH or len(train_loader), len(train_loader))
    student, kd = build_kd(cfg, student, iters_per_epoch)
    logger.info("Student: {} ({:.1f}M params incl. KD head), randomly initialised: {}".format(
        cfg.MODEL.TRANSFORMER_TYPE, sum(p.numel() for p in student.parameters()) / 1e6,
        cfg.MODEL.PRETRAIN_CHOICE != 'imagenet'))

    reid_loss_fn, _ = make_loss(cfg, num_classes=num_classes)
    optimizer = make_kd_optimizer(cfg, student)
    scheduler = create_scheduler(cfg, optimizer)

    do_train_kd(cfg, student, teacher, kd, train_loader, train_loader_normal, val_loader, optimizer, scheduler,
                reid_loss_fn, num_query, args.local_rank)

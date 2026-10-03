from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
import datetime
import faulthandler
import signal
# from timm.scheduler import create_scheduler
from config import cfg

# `kill -USR1 <pid>` prints every Python thread's stack to stderr (py-spy can't attach on this node).
faulthandler.register(signal.SIGUSR1, all_threads=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # torchrun passes the rank via the LOCAL_RANK env var; torch.distributed.launch passes --local_rank.
    parser.add_argument("--local_rank", "--local-rank", default=int(os.environ.get("LOCAL_RANK", 0)), type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Offset by rank so each DDP process draws different augmentations; rank 0 (and single-GPU) keeps
    # cfg.SOLVER.SEED. Model weights stay identical across ranks because DDP broadcasts rank 0's on wrap.
    set_seed(cfg.SOLVER.SEED + int(os.environ.get("RANK", 0)))

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True, resume=cfg.SOLVER.RESUME)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        # Must exceed rank 0's full evaluation, during which the other ranks wait at a barrier. A rank that
        # hangs mid-step is only detected (and the job killed, so it can be resumed) after this long.
        timeout_min = int(os.environ.get('DIST_TIMEOUT_MIN', 120))
        torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                             timeout=datetime.timedelta(minutes=timeout_min))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )

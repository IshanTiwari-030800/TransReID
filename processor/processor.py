import logging
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

CKPT_NAME = 'checkpoint_latest.pth'



def _wandb_log(data, step):
    """wandb.log at an explicit step, skipping steps the run already holds (see the resume note in do_train)."""
    # starting_step (last logged step + 1 on a resumed run, 0 on a fresh one) is set at init; run.step reads 0
    # right after a resume, so it can't be used for this.
    if step >= wandb.run.starting_step:
        wandb.log(data, step=step)

def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


def _save_checkpoint(path, epoch, global_step, best_map, wandb_run_id, model, center_criterion,
                     optimizer, optimizer_center, scaler):
    """Full training state, written atomically so a crash mid-write can't corrupt the last good one."""
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'best_map': best_map,
        'wandb_run_id': wandb_run_id,
        'model': _unwrap(model).state_dict(),
        'center_criterion': center_criterion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_center': optimizer_center.state_dict(),
        'scaler': scaler.state_dict(),
        'rng': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        },
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):

    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    best_map = 0.0

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')

    _LOCAL_PROCESS_GROUP = None

    is_main_process = (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0

    scaler = amp.GradScaler()
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, CKPT_NAME)
    start_epoch = 1
    global_step = 0
    wandb_run_id = None
    # Params must already be on their GPU when the optimizer state is loaded: load_state_dict casts
    # momentum buffers to each param's *current* device, so loading while on CPU strands them there.
    model.to(local_rank)
    if cfg.SOLVER.RESUME and os.path.exists(ckpt_path):
        # Load before DDP wrapping (checkpoint keys have no 'module.' prefix).
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        center_criterion.load_state_dict(ckpt['center_criterion'])
        optimizer.load_state_dict(ckpt['optimizer'])
        optimizer_center.load_state_dict(ckpt['optimizer_center'])
        scaler.load_state_dict(ckpt['scaler'])
        torch.set_rng_state(ckpt['rng']['torch'])
        torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
        np.random.set_state(ckpt['rng']['numpy'])
        random.setstate(ckpt['rng']['python'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_map = ckpt['best_map']
        wandb_run_id = ckpt['wandb_run_id']
        logger.info("Resumed from {} (finished epoch {}); continuing at epoch {}".format(
            ckpt_path, ckpt['epoch'], start_epoch))
        del ckpt
    elif cfg.SOLVER.RESUME:
        logger.info("SOLVER.RESUME is on but {} does not exist; starting from scratch".format(ckpt_path))

    if is_main_process:
        wandb.init(
            project=cfg.WANDB.PROJECT,
            name=cfg.WANDB.NAME or os.path.basename(os.path.normpath(cfg.OUTPUT_DIR)),
            config=yaml.safe_load(cfg.dump()),
            id=wandb_run_id,           # None on a fresh run -> wandb generates one
            resume="allow",
        )
        wandb_run_id = wandb.run.id
        # A resume restarts from the last epoch checkpoint, but the interrupted job may already have logged
        # steps past it, and wandb drops any step below its current one (with a warning per call). Rewinding
        # the run would drop those instead, but that is a private-preview wandb feature. The resumed job
        # replays the same batches with the same RNG state, so the values already logged for those steps
        # are the same ones it would log; skip re-logging them.
        if wandb.run.starting_step > global_step + 1:
            logger.info("wandb run already has steps up to {} (logged before the interruption); "
                        "resuming at step {}, logging restarts after that".format(
                            wandb.run.starting_step - 1, global_step))
        # Patch-embedding architecture details, read off the actual instantiated backbone
        # (not the model.to(local_rank)/DDP-wrapped `model`, so this must run before that below).
        backbone = model.base
        patch_embed = backbone.patch_embed
        wandb.config.update({
            "arch/patch_size": tuple(patch_embed.patch_size),
            "arch/stride_size": tuple(cfg.MODEL.STRIDE_SIZE),
            "arch/num_patches_y": patch_embed.num_y,
            "arch/num_patches_x": patch_embed.num_x,
            "arch/num_patches": patch_embed.num_y * patch_embed.num_x,
            "arch/pos_embed_shape": tuple(backbone.pos_embed.shape),
        })

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # train
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):

            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            global_step += 1
            if (n_iter + 1) % log_period == 0:
                base_lr = scheduler._get_lr(epoch)[0]
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, base_lr))
                if is_main_process:
                    _wandb_log({
                        "train/loss": loss_meter.avg,
                        "train/acc": acc_meter.avg,
                        "train/lr": base_lr,
                        "epoch": epoch,
                    }, step=global_step)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        # IMS_PER_BATCH is the global batch (summed over GPUs under DDP), so this is total throughput.
        samples_per_sec = cfg.SOLVER.IMS_PER_BATCH / time_per_batch
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch, samples_per_sec))
        if is_main_process:
            _wandb_log({
                "epoch": epoch,
                "train/epoch_loss": loss_meter.avg,
                "train/epoch_acc": acc_meter.avg,
                "train/time_per_batch": time_per_batch,
                "train/samples_per_sec": samples_per_sec,
                }, step=global_step)

        if is_main_process:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_latest.pth'))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    # Forward through the unwrapped module: a DDP forward can issue a buffer broadcast
                    # (BN running stats), a collective the other ranks never join -> NCCL desync/hang.
                    eval_model = _unwrap(model)
                    eval_model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = eval_model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    if is_main_process:
                        _wandb_log({
                            "epoch": epoch,
                            "eval/mAP": mAP,
                            "eval/rank1": cmc[0],
                            "eval/rank5": cmc[4],
                            "eval/rank10": cmc[9],
                    }, step=global_step)
                        if mAP > best_map:
                            best_map = mAP
                            torch.save(model.state_dict(),
                                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                            logger.info("New best eval mAP: {:.1%} (epoch {}) -> saved {}_best.pth".format(
                                best_map, epoch, cfg.MODEL.NAME))
                    torch.cuda.empty_cache()
                # Other ranks wait here for rank 0's eval instead of racing into the next epoch's collectives.
                dist.barrier()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                if is_main_process:
                    _wandb_log({
                        "epoch": epoch,
                        "eval/mAP": mAP,
                        "eval/rank1": cmc[0],
                        "eval/rank5": cmc[4],
                        "eval/rank10": cmc[9],
                    }, step=global_step)
                    if mAP > best_map:
                        best_map = mAP
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                        logger.info("New best eval mAP: {:.1%} (epoch {}) -> saved {}_best.pth".format(
                            best_map, epoch, cfg.MODEL.NAME))
                torch.cuda.empty_cache()

        # Last thing in the epoch (after eval, so best_map is current): everything needed to resume.
        if is_main_process:
            _save_checkpoint(os.path.join(cfg.OUTPUT_DIR, CKPT_NAME), epoch, global_step, best_map,
                             wandb_run_id, model, center_criterion, optimizer, optimizer_center, scaler)

    if is_main_process:
        wandb.finish()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):

    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]



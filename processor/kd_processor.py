import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.cuda import amp

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from .processor import CKPT_NAME, _unwrap, _wandb_log


def _reid_model(model):
    """The bare ReID model (what test.py loads), inside any DDP / StudentWithHead wrapping."""
    m = _unwrap(model)
    return getattr(m, 'reid', m)


@torch.no_grad()
def evaluate(model, val_loader, num_query, feat_norm):
    model.eval()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=feat_norm)
    evaluator.reset()
    for img, vid, camid, camids, target_view, _ in val_loader:
        feat = model(img.cuda(), cam_label=camids.cuda(), view_label=target_view.cuda())
        evaluator.update((feat, vid, camid))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    return cmc, mAP


def _save_checkpoint(path, epoch, global_step, best_map, best_epoch, wandb_run_id, model, kd, optimizer, scaler):
    """Full training state, written atomically so a crash mid-write can't corrupt the last good one."""
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'best_map': best_map,
        'best_epoch': best_epoch,
        'wandb_run_id': wandb_run_id,
        'model': _unwrap(model).state_dict(),
        'kd': kd.state_dict(),
        'optimizer': optimizer.state_dict(),
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


def _append_metrics(output_dir, record):
    with open(os.path.join(output_dir, 'metrics.jsonl'), 'a') as f:
        f.write(json.dumps(record) + '\n')


def do_train_kd(cfg, model, teacher, kd, train_loader, train_loader_normal, val_loader, optimizer, scheduler,
                reid_loss_fn, num_query, local_rank):
    """Train the student with the ReID losses plus the KD term from `kd` (see kd/methods.py).

    Saves to OUTPUT_DIR: transformer_best.pth / transformer_latest.pth (student ReID weights only, loadable by
    test.py), checkpoint_latest.pth (full state for SOLVER.RESUME), metrics.jsonl (every evaluation).
    """
    logger = logging.getLogger("transreid.train")
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS
    max_iters = cfg.SOLVER.MAX_ITERS_PER_EPOCH
    is_main = (not cfg.MODEL.DIST_TRAIN) or dist.get_rank() == 0
    use_wandb = cfg.WANDB.ENABLED and is_main
    student_views, teacher_views = kd.student_views, kd.teacher_views

    scaler = amp.GradScaler()
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, CKPT_NAME)
    start_epoch, global_step, best_map, best_epoch, wandb_run_id = 1, 0, 0.0, 0, None
    # On the GPU before the optimizer state is loaded (load_state_dict puts its buffers on each param's device).
    model.to(local_rank)
    kd.to(local_rank)
    if teacher is not None:
        teacher.to(local_rank)
    if cfg.SOLVER.RESUME and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        kd.load_state_dict(ckpt['kd'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        torch.set_rng_state(ckpt['rng']['torch'])
        torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
        np.random.set_state(ckpt['rng']['numpy'])
        random.setstate(ckpt['rng']['python'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_map, best_epoch = ckpt['best_map'], ckpt['best_epoch']
        wandb_run_id = ckpt['wandb_run_id']
        logger.info("Resumed from {} (finished epoch {}); continuing at epoch {}".format(
            ckpt_path, ckpt['epoch'], start_epoch))
        del ckpt
    elif cfg.SOLVER.RESUME:
        logger.info("SOLVER.RESUME is on but {} does not exist; starting from scratch".format(ckpt_path))
    if start_epoch > epochs:
        # A chained Slurm job that starts after the run already finished.
        logger.info("All {} epochs already done (best mAP {:.1%} at epoch {}); nothing to do".format(
            epochs, best_map, best_epoch))
        return

    if use_wandb:
        wandb.init(project=cfg.WANDB.PROJECT, name=cfg.WANDB.NAME or os.path.basename(os.path.normpath(cfg.OUTPUT_DIR)),
                   config=yaml.safe_load(cfg.dump()), id=wandb_run_id, resume="allow")
        wandb_run_id = wandb.run.id

    if teacher is not None and cfg.KD.TEACHER.EVAL_AT_START and start_epoch == 1:
        if is_main:
            cmc, mAP = evaluate(teacher.model, val_loader, num_query, cfg.TEST.FEAT_NORM)
            logger.info("Teacher on val: mAP {:.1%}, Rank-1 {:.1%}, Rank-5 {:.1%}".format(mAP, cmc[0], cmc[4]))
            _append_metrics(cfg.OUTPUT_DIR, {'model': 'teacher', 'mAP': float(mAP), 'rank1': float(cmc[0]),
                                             'rank5': float(cmc[4]), 'rank10': float(cmc[9])})
            if use_wandb:
                wandb.run.summary.update({'teacher/mAP': mAP, 'teacher/rank1': cmc[0]})
            torch.cuda.empty_cache()
        if cfg.MODEL.DIST_TRAIN:
            dist.barrier()

    if start_epoch == 1:
        kd.prepare(model, teacher, train_loader_normal, cfg.MODEL.DIST_TRAIN)

    if cfg.MODEL.DIST_TRAIN:
        # find_unused_parameters: TransReID's ImageNet `fc` head is never used in the forward pass.
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          find_unused_parameters=True)
    params = [p for p in model.parameters() if p.requires_grad]

    meters = {k: AverageMeter() for k in ('loss', 'reid_loss', 'kd_loss', 'acc', 'grad_norm')}
    stat_meters = {}
    logger.info('start training: KD method {}, student sees {} view(s), teacher sees {}'.format(
        cfg.KD.METHOD, student_views, teacher_views))
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        for m in list(meters.values()) + list(stat_meters.values()):
            m.reset()
        scheduler.step(epoch)
        model.train()
        n_iters = len(train_loader) if not max_iters else min(max_iters, len(train_loader))
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            if n_iter >= n_iters:
                break
            img = img.cuda(non_blocking=True)
            views = list(img.unbind(1)) if img.dim() == 5 else [img]
            target, cam, view = vid.cuda(), target_cam.cuda(), target_view.cuda()
            s_img = torch.cat(views[:student_views])
            s_target, s_cam, s_view = (t.repeat(student_views) for t in (target, cam, view))
            with_stats = (n_iter + 1) % log_period == 0

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                out = model(s_img, s_target, cam_label=s_cam, view_label=s_view)
                s_out = {'score': out[0], 'global_feat': out[1]}
                if len(out) > 2:
                    s_out['head'] = out[2]
                reid_loss = reid_loss_fn(s_out['score'], s_out['global_feat'], s_target, s_cam)
                t_out = None
                if teacher is not None:
                    t_out = teacher(torch.cat(views[:teacher_views]), cam_label=cam.repeat(teacher_views),
                                    view_label=view.repeat(teacher_views))
                kd_loss, kd_stats = kd(s_out, t_out, target, epoch, with_stats)
                loss = reid_loss + kd_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            kd.before_step(_unwrap(model), epoch)
            clip = cfg.SOLVER.CLIP_GRAD if cfg.SOLVER.CLIP_GRAD > 0 else float('inf')
            grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
            scaler.step(optimizer)
            scaler.update()
            kd.after_step(_unwrap(model), global_step)
            global_step += 1

            n = s_img.shape[0]
            meters['loss'].update(loss.item(), n)
            meters['reid_loss'].update(reid_loss.item(), n)
            meters['kd_loss'].update(kd_loss.item(), n)
            meters['acc'].update((s_out['score'].argmax(1) == s_target).float().mean().item(), 1)
            if torch.isfinite(grad_norm):
                meters['grad_norm'].update(grad_norm.item(), 1)
            for k, v in kd_stats.items():
                stat_meters.setdefault(k, AverageMeter()).update(float(v), 1)

            if with_stats:
                base_lr = scheduler._get_lr(epoch)[0]
                stats_str = ', '.join('{} {:.4g}'.format(k.split('/')[-1], m.avg) for k, m in stat_meters.items())
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (reid {:.3f}, kd {:.3f}), Acc: {:.3f}, "
                            "Grad: {:.2f}, Base Lr: {:.2e} | {}".format(
                                epoch, n_iter + 1, n_iters, meters['loss'].avg, meters['reid_loss'].avg,
                                meters['kd_loss'].avg, meters['acc'].avg, meters['grad_norm'].avg, base_lr,
                                stats_str))
                if use_wandb:
                    log = {'train/' + k: m.avg for k, m in meters.items()}
                    log.update({k: m.avg for k, m in stat_meters.items()})
                    log.update({'train/lr': base_lr, 'epoch': epoch})
                    _wandb_log(log, step=global_step)

        time_per_batch = (time.time() - start_time) / n_iters
        samples_per_sec = cfg.SOLVER.IMS_PER_BATCH / time_per_batch
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s] Peak mem: {:.1f}GB".format(
            epoch, time_per_batch, samples_per_sec, torch.cuda.max_memory_allocated() / 2 ** 30))

        if is_main:
            torch.save(_reid_model(model).state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_latest.pth'))
            if use_wandb:
                _wandb_log({'epoch': epoch, 'train/epoch_loss': meters['loss'].avg,
                            'train/time_per_batch': time_per_batch, 'train/samples_per_sec': samples_per_sec},
                           step=global_step)

        if epoch % eval_period == 0 or epoch == epochs:
            if is_main:
                # Forward through the unwrapped module: a DDP forward can issue a buffer broadcast that the other
                # ranks (waiting at the barrier) never join.
                cmc, mAP = evaluate(_reid_model(model), val_loader, num_query, cfg.TEST.FEAT_NORM)
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                _append_metrics(cfg.OUTPUT_DIR, {'model': 'student', 'epoch': epoch, 'mAP': float(mAP),
                                                 'rank1': float(cmc[0]), 'rank5': float(cmc[4]),
                                                 'rank10': float(cmc[9])})
                if use_wandb:
                    _wandb_log({'epoch': epoch, 'eval/mAP': mAP, 'eval/rank1': cmc[0], 'eval/rank5': cmc[4],
                                'eval/rank10': cmc[9]}, step=global_step)
                if mAP > best_map:
                    best_map, best_epoch = mAP, epoch
                    torch.save(_reid_model(model).state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                    logger.info("New best eval mAP: {:.1%} (epoch {}) -> saved {}_best.pth".format(
                        best_map, epoch, cfg.MODEL.NAME))
                torch.cuda.empty_cache()
            if cfg.MODEL.DIST_TRAIN:
                dist.barrier()

        # Last thing in the epoch (after eval, so best_map is current): everything needed to resume.
        if is_main:
            _save_checkpoint(ckpt_path, epoch, global_step, best_map, best_epoch, wandb_run_id, model, kd,
                             optimizer, scaler)

    logger.info("Training done. Best mAP {:.1%} at epoch {}".format(best_map, best_epoch))
    if use_wandb:
        wandb.run.summary.update({'best/mAP': best_map, 'best/epoch': best_epoch})
        wandb.finish()

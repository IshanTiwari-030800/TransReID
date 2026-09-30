"""
Find the largest training batch size that fits on this GPU for a given config,
by running real forward+backward+optimizer.step() steps (same AMP path as
processor.do_train) on synthetic batches, increasing batch size until CUDA OOM.

Usage:
    python probe_batch_size.py --config_file configs/VeRi/vit_base_p12.yml MODEL.DEVICE_ID "('0')"
"""
import argparse
import sys

import torch
from torch.cuda import amp

from config import cfg
from datasets import make_dataloader
from model import make_model
from loss import make_loss
from solver import make_optimizer


def try_batch_size(model, loss_fn, optimizer, scaler, batch_size, img_size, num_classes, cam_num, view_num, num_instance, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Mimic RandomIdentitySampler's PK structure: each identity appears exactly
    # `num_instance` times, since triplet hard-mining assumes a fixed positives-per-row count.
    assert batch_size % num_instance == 0, "batch_size must be a multiple of NUM_INSTANCE"
    num_ids = batch_size // num_instance
    ids = torch.randint(0, num_classes, (num_ids,))
    target = ids.repeat_interleave(num_instance).to(device)

    img = torch.randn(batch_size, 3, *img_size, device=device)
    target_cam = torch.randint(0, max(cam_num, 1), (batch_size,), device=device)
    target_view = torch.randint(0, max(view_num, 1), (batch_size,), device=device)

    try:
        with amp.autocast(enabled=True):
            score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
            loss = loss_fn(score, feat, target, target_cam)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--min_bs", type=int, default=4)
    parser.add_argument("--max_bs", type=int, default=128)
    parser.add_argument("--step", type=int, default=4, help="should match DATALOADER.NUM_INSTANCE")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not torch.cuda.is_available():
        print("No CUDA device available.")
        sys.exit(1)

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}, total mem: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")

    # Real dataloader build gives us the true num_classes / cam_num / view_num for this
    # dataset+config combo, so the classifier / SIE embedding sizes match a real training run.
    _, _, _, _, num_classes, cam_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num=view_num).to(device)
    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    scaler = amp.GradScaler()

    img_size = cfg.INPUT.SIZE_TRAIN
    step = args.step
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    last_good = None

    for bs in range(args.min_bs, args.max_bs + 1, step):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ok = try_batch_size(model, loss_fn, optimizer, scaler, bs, img_size, num_classes, cam_num, view_num, num_instance, device)
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        if ok:
            print(f"batch_size={bs:4d}  OK    peak_mem={peak_gib:.2f} GiB")
            last_good = bs
        else:
            print(f"batch_size={bs:4d}  OOM")
            break

    print()
    if last_good is None:
        print(f"Even batch_size={args.min_bs} OOM'd on this GPU with this config.")
        sys.exit(1)

    safe_bs = max(step, (int(last_good * 0.9) // step) * step)
    print(f"Largest batch size that fit: {last_good}")
    print(f"Recommended IMS_PER_BATCH (~90% headroom, rounded to a multiple of {step}): {safe_bs}")


if __name__ == "__main__":
    main()

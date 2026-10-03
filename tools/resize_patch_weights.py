"""Make an ImageNet ViT-B checkpoint for a different patch size from the ViT-B/16 one.

This is the recipe behind pretrained_weights/imagenet_vit8.pth and imagenet_vit12.pth (`--check` reproduces both):
  - patch_embed.proj.weight: each 16x16 filter is resized bicubically to the new patch size, then each of the
    768 output filters is rescaled back to its original L2 norm;
  - pos_embed: the cls token is kept and the 14x14 grid is resized bicubically to GRID x GRID, where
    GRID = 256 // max(patch) (square, because TransReID's resize_pos_embed assumes the stored grid is square;
    it resizes it to the model's actual token grid when loading);
  - head.* (the ImageNet classifier) is dropped and fc.* is a fresh init (trunc-normal std 0.02, zero bias), as
    in the existing files, which hold the TransReID backbone's own randomly initialised fc. TransReID never calls
    that layer in forward(), so its values don't affect training;
  - every other tensor is copied unchanged.

Usage:
  python tools/resize_patch_weights.py --patch 8 4 --out /mnt/data/ishant/pretrained_weights/imagenet_vit8x4.pth
  python tools/resize_patch_weights.py --check
"""
import argparse
from collections import OrderedDict

import torch
import torch.nn.functional as F

WEIGHTS_DIR = '/mnt/data/ishant/pretrained_weights'
SRC = f'{WEIGHTS_DIR}/jx_vit_base_p16_224-80ecf9dd.pth'


def resize_checkpoint(src, patch, grid=None):
    ph, pw = patch
    grid = grid or 256 // max(ph, pw)
    gen = torch.Generator().manual_seed(0)
    out = OrderedDict()
    for k, v in src.items():
        if k == 'patch_embed.proj.weight':
            w = F.interpolate(v, size=(ph, pw), mode='bicubic', align_corners=False)
            v = w * (v.flatten(1).norm(dim=1) / w.flatten(1).norm(dim=1)).view(-1, 1, 1, 1)
        elif k == 'pos_embed':
            cls, pos = v[:, :1], v[:, 1:]
            old = int(pos.shape[1] ** 0.5)
            pos = pos.reshape(1, old, old, -1).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(grid, grid), mode='bicubic', align_corners=False)
            v = torch.cat([cls, pos.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)], dim=1)
        elif k == 'head.weight':
            k, v = 'fc.weight', torch.nn.init.trunc_normal_(torch.empty_like(v), std=.02, generator=gen)
        elif k == 'head.bias':
            k, v = 'fc.bias', torch.zeros_like(v)
        out[k] = v
    return out


def check():
    src = torch.load(SRC, map_location='cpu')
    for p in (8, 12):
        ref = torch.load(f'{WEIGHTS_DIR}/imagenet_vit{p}.pth', map_location='cpu')
        new = resize_checkpoint(src, (p, p))
        assert set(new) == set(ref), set(new) ^ set(ref)
        assert all(new[k].shape == ref[k].shape for k in ref)
        worst = max(((new[k] - ref[k]).norm() / ref[k].norm()).item() for k in ref if not k.startswith('fc.'))
        fc_std = (new['fc.weight'].std().item(), ref['fc.weight'].std().item())
        print(f'imagenet_vit{p}.pth: same keys and shapes; worst relative difference {worst:.1e} '
              f'(fc excluded: random init, std {fc_std[0]:.4f} here vs {fc_std[1]:.4f})')
        assert worst < 1e-5 and not ref['fc.bias'].any()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch', type=int, nargs=2, metavar=('H', 'W'))
    ap.add_argument('--out')
    ap.add_argument('--check', action='store_true', help='reproduce imagenet_vit8.pth and imagenet_vit12.pth')
    args = ap.parse_args()
    if args.check:
        check()
    else:
        ckpt = resize_checkpoint(torch.load(SRC, map_location='cpu'), args.patch)
        torch.save(ckpt, args.out)
        print('saved', args.out, '| patch_embed', tuple(ckpt['patch_embed.proj.weight'].shape),
              '| pos_embed', tuple(ckpt['pos_embed'].shape))

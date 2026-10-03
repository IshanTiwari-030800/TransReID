""" Deformable ViT for TransReID

A plain ViT-Base (CLS token, absolute position embeddings, 12 blocks) whose self-attention is replaced by
deformable attention in the style of DAT (Xia et al., "Vision Transformer with Deformable Attention", CVPR 2022).

In a deformable block every token is still a query, but the keys/values are not the full token grid. Instead:
  1. a reference grid is laid over the token map, DOWNSAMPLE times coarser than it;
  2. a small offset network (depthwise conv -> LayerNorm -> GELU -> 1x1 conv) reads the queries and predicts a
     2-D offset for every reference point, per group of channels;
  3. features are bilinearly sampled at reference + offset, and keys/values are projected from those samples.
The CLS token is always kept as an extra key/value. With 8x8 patches on a 256x256 image (32x32 tokens) and
DOWNSAMPLE 2, each query attends to 1 + 16x16 = 257 keys instead of 1025.

Differences from DAT: DAT is a 4-stage hierarchy and alternates local-window and deformable attention; here the
model stays a single-stage ViT so the TransReID head (CLS feature -> BNNeck) and evaluation are unchanged.
DAT's relative position bias is also omitted: tokens carry absolute position embeddings from the input layer.
The last conv of the offset network is zero-initialised, so training starts from uniform-grid sampling.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_pytorch import TransReID


class LayerNormChannels(nn.Module):
    """LayerNorm over the channel dim of a (B, C, H, W) map."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DeformableAttention(nn.Module):
    def __init__(self, dim, num_heads, grid_size, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 groups=4, downsample=2, kernel=5, offset_range=2.0):
        super().__init__()
        assert dim % num_heads == 0 and dim % groups == 0, (dim, num_heads, groups)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.groups = groups
        self.grid_size = tuple(grid_size)  # (H, W) of the patch-token map

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        group_dim = dim // groups
        self.conv_offset = nn.Sequential(
            nn.Conv2d(group_dim, group_dim, kernel, stride=downsample, padding=kernel // 2, groups=group_dim),
            LayerNormChannels(group_dim),
            nn.GELU(),
            nn.Conv2d(group_dim, 2, 1, bias=False),
        )
        nn.init.zeros_(self.conv_offset[-1].weight)  # start from uniform-grid sampling

        H, W = self.grid_size
        Hk, Wk = self._key_grid(H, W, kernel, downsample)
        self.key_grid_size = (Hk, Wk)
        # Reference points at the centres of the Hk x Wk cells, in grid_sample's normalised [-1, 1] coordinates
        # (align_corners=False), stored as (y, x) to match the offset channels.
        ref_y = (torch.arange(Hk, dtype=torch.float32) + 0.5) / Hk * 2 - 1
        ref_x = (torch.arange(Wk, dtype=torch.float32) + 0.5) / Wk * 2 - 1
        ref = torch.stack(torch.meshgrid(ref_y, ref_x, indexing='ij'), dim=0)  # (2, Hk, Wk)
        self.register_buffer('reference', ref.unsqueeze(0), persistent=False)
        # tanh(offset) is scaled to +-offset_range reference cells (a cell is 2/Hk tall, 2/Wk wide)
        self.register_buffer('offset_scale',
                             torch.tensor([2.0 / Hk, 2.0 / Wk]).reshape(1, 2, 1, 1) * offset_range,
                             persistent=False)

    @staticmethod
    def _key_grid(H, W, kernel, stride):
        pad = kernel // 2
        return (H + 2 * pad - kernel) // stride + 1, (W + 2 * pad - kernel) // stride + 1

    def sampling_points(self, q_patch):
        """(B, N, C) patch queries -> (B*groups, 2, Hk, Wk) sampling positions in (y, x), normalised to [-1, 1]."""
        B, N, C = q_patch.shape
        H, W = self.grid_size
        q_map = q_patch.transpose(1, 2).reshape(B * self.groups, C // self.groups, H, W)
        offset = self.conv_offset(q_map).float().tanh() * self.offset_scale
        return (self.reference + offset).clamp(-1, 1)

    def forward(self, x):
        B, T, C = x.shape
        H, W = self.grid_size
        assert T == H * W + 1, f"expected {H}x{W} patch tokens + CLS, got {T} tokens"
        heads, groups = self.num_heads, self.groups

        q = self.q(x)
        pos = self.sampling_points(q[:, 1:])

        # Each group of channels samples the patch-token map at its own points.
        patch_map = x[:, 1:].transpose(1, 2).reshape(B * groups, C // groups, H, W)
        grid = pos.permute(0, 2, 3, 1).flip(-1).to(patch_map.dtype)  # grid_sample wants (x, y)
        sampled = F.grid_sample(patch_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampled = sampled.reshape(B, C, -1).transpose(1, 2)  # (B, Hk*Wk, C)

        kv_in = torch.cat([x[:, :1], sampled], dim=1)
        k, v = self.k(kv_in), self.v(kv_in)
        q, k, v = (t.reshape(B, t.shape[1], heads, C // heads).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.,
                                             scale=self.scale)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


def deformable_vit_base_patch8_TransReID(img_size=(256, 128), stride_size=8, drop_rate=0.0, attn_drop_rate=0.0,
                                         drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5,
                                         deform_blocks=(), deform_groups=4, deform_downsample=2, deform_kernel=5,
                                         deform_offset_range=2.0, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=8, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=sie_xishu,
        local_feature=local_feature, **kwargs)

    grid_size = (model.patch_embed.num_y, model.patch_embed.num_x)
    blocks = list(deform_blocks) or range(len(model.blocks))
    for i in blocks:
        attn = DeformableAttention(768, 12, grid_size, qkv_bias=True, attn_drop=attn_drop_rate, proj_drop=drop_rate,
                                   groups=deform_groups, downsample=deform_downsample, kernel=deform_kernel,
                                   offset_range=deform_offset_range)
        attn.apply(model._init_weights)  # q/k/v/proj like the rest of the ViT; leaves the zeroed offset conv alone
        model.blocks[i].attn = attn
    print('deformable attention in blocks {}: {}x{} tokens -> {}x{} sampled keys, {} offset groups'.format(
        list(blocks), *grid_size, *attn.key_grid_size, deform_groups))
    return model

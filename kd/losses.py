import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _all_reduce(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t)
    return t


def _world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


# --------------------------------------------------------------------------------------------------------------
# Soft KD
# --------------------------------------------------------------------------------------------------------------

def soft_kd_loss(student_logits, teacher_logits, temperature):
    """Hinton et al. 2015: KL(teacher || student) on temperature-softened logits, scaled by T^2 so its gradient
    magnitude doesn't shrink as T grows."""
    T = temperature
    log_p_s = F.log_softmax(student_logits.float() / T, dim=-1)
    log_p_t = F.log_softmax(teacher_logits.float() / T, dim=-1)
    return F.kl_div(log_p_s, log_p_t, reduction='batchmean', log_target=True) * T * T


# --------------------------------------------------------------------------------------------------------------
# DINO
# --------------------------------------------------------------------------------------------------------------

class DINOHead(nn.Module):
    """DINO projection head: 3-layer MLP -> L2-normalised bottleneck -> similarity to out_dim prototypes.

    The prototypes are L2-normalised in the forward pass, which is DINO's weight-normalised last layer with its
    gain frozen at 1 (norm_last_layer=True). Outputs are cosine similarities in [-1, 1]; sinkhorn_knopp relies
    on that bound.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.prototypes = nn.Linear(bottleneck_dim, out_dim, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.normalize(self.mlp(x).float(), dim=-1)
        return F.linear(x, F.normalize(self.prototypes.weight.float(), dim=-1))


@torch.no_grad()
def sinkhorn_knopp(logits, temperature, n_iters=3):
    """Balanced soft assignment of the batch to the prototypes (SwAV / DINOv2 teacher centering).

    Alternately normalises prototype marginals to 1/K and sample marginals to 1/B, so every prototype receives
    the same total mass from the (global, all-ranks) batch. A teacher that maps everything to one prototype is
    then impossible by construction, which is what prevents collapse. logits: (B, K) cosine similarities in
    [-1, 1]. Returns (B, K) rows that sum to 1.
    """
    # Subtracting the bound 1 (not the per-rank max) keeps exp() <= 1 and is the same constant on every rank,
    # so it cancels in the first normalisation.
    Q = torch.exp((logits.float() - 1) / temperature).t()  # (K, B)
    K, B = Q.shape[0], Q.shape[1] * _world_size()
    Q /= _all_reduce(Q.sum())
    for _ in range(n_iters):
        Q /= _all_reduce(Q.sum(dim=1, keepdim=True))
        Q /= K
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= B
    Q *= B
    return Q.t()


class EMACentering(nn.Module):
    """DINO v1 teacher centering: subtract an EMA of the mean teacher output before the sharpened softmax."""
    def __init__(self, out_dim, momentum):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('center', torch.zeros(1, out_dim))

    @torch.no_grad()
    def forward(self, logits, temperature):
        probs = F.softmax((logits.float() - self.center) / temperature, dim=-1)
        batch_center = _all_reduce(logits.float().sum(dim=0, keepdim=True)) / (logits.shape[0] * _world_size())
        self.center.mul_(self.momentum).add_(batch_center, alpha=1 - self.momentum)
        return probs


def dino_cross_entropy(student_logits, teacher_probs, student_temp, same_view_pairs):
    """Mean over (student view i, teacher view j) pairs of H(teacher_j, softmax(student_i / student_temp))."""
    total, n_pairs = 0., 0
    for i, s in enumerate(student_logits):
        log_p = F.log_softmax(s.float() / student_temp, dim=-1)
        for j, q in enumerate(teacher_probs):
            if i == j and not same_view_pairs:
                continue
            total = total + torch.sum(-q * log_p, dim=-1).mean()
            n_pairs += 1
    return total / n_pairs


def koleo_loss(feats, labels=None, eps=1e-8):
    """KoLeo regulariser (Sablayrolles et al. 2019, as used in DINOv2): -mean log distance to the nearest
    neighbour in the batch, on L2-normalised features. Spreads features uniformly over the sphere, countering
    dimensional collapse.

    With labels, neighbours with the same label are skipped: in a P x K identity-sampled ReID batch the nearest
    sample is usually the same identity, and pushing those apart would work against the triplet loss.
    """
    x = F.normalize(feats.float(), dim=-1, eps=eps)
    sims = x @ x.t()
    if labels is None:
        excluded = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    else:
        excluded = labels[:, None] == labels[None, :]
    nn_idx = sims.masked_fill(excluded, -2.).argmax(dim=1)
    dist_nn = F.pairwise_distance(x, x[nn_idx], eps=eps)
    return -torch.log(dist_nn + eps).mean()


def entropy(probs, eps=1e-12):
    return -(probs * torch.log(probs + eps)).sum(dim=-1)

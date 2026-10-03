import logging

import torch
import torch.nn.functional as F


@torch.no_grad()
def teacher_features(teacher, loader):
    """L2-normalised teacher global features for every image in `loader` (a val-style loader, no augmentation)."""
    feats = []
    for img, _, _, camids, target_view, _ in loader:
        with torch.autocast('cuda'):
            f = teacher(img.cuda(), cam_label=camids.cuda(), view_label=target_view.cuda())['global_feat']
        feats.append(F.normalize(f.float(), dim=-1))
    return torch.cat(feats)


@torch.no_grad()
def kmeans_plusplus(x, k, g):
    """k-means++ seeding with cosine distance: each new seed is drawn with probability proportional to its squared
    distance from the nearest seed so far, so seeds spread over the data instead of doubling up in dense regions."""
    n = x.shape[0]
    idx = [int(torch.randint(n, (1,), generator=g))]
    min_dist = (1 - x @ x[idx[0]]).clamp_(min=0)
    for _ in range(1, k):
        probs = (min_dist ** 2).cpu()
        i = int(torch.multinomial(probs / probs.sum(), 1, generator=g)) if probs.sum() > 0 else \
            int(torch.randint(n, (1,), generator=g))
        idx.append(i)
        torch.minimum(min_dist, (1 - x @ x[i]).clamp_(min=0), out=min_dist)
    return x[idx].clone()


@torch.no_grad()
def spherical_kmeans(x, k, iters=30, seed=0, chunk=65536):
    """k-means with cosine similarity on unit vectors x (N, D), k-means++ seeded. Returns unit centroids (k, D) and
    the mean cosine similarity of each point to its centroid. Empty clusters are re-seeded with random points."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    n = x.shape[0]
    if k > n:
        raise ValueError('cannot make {} prototypes from {} teacher features; lower KD.DINO.OUT_DIM'.format(k, n))
    centroids = kmeans_plusplus(x, k, g)
    for _ in range(iters):
        assign = torch.cat([(x[i:i + chunk] @ centroids.t()).argmax(1) for i in range(0, n, chunk)])
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k)
        empty = counts == 0
        if empty.any():
            sums[empty] = x[torch.randint(n, (int(empty.sum()),), generator=g).to(x.device)]
        centroids = F.normalize(sums, dim=-1)
    sims = torch.cat([(x[i:i + chunk] @ centroids.t()).max(1).values for i in range(0, n, chunk)])
    return centroids, sims.mean().item()


def kmeans_prototypes(teacher, loader, k, iters, seed):
    logger = logging.getLogger("transreid.train")
    feats = teacher_features(teacher, loader)
    centroids, mean_sim = spherical_kmeans(feats, k, iters, seed)
    logger.info("DINO prototypes: spherical k-means of {} teacher features into {} clusters ({} iters), "
                "mean cosine to centroid {:.3f}".format(feats.shape[0], k, iters, mean_sim))
    return centroids

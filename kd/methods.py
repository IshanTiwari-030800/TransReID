"""The KD recipes. Each exposes the same interface to processor/kd_processor.py:

    student_views / teacher_views    how many of the batch's augmented views each network sees
    forward(s_out, t_out, labels, epoch, with_stats) -> (weighted KD loss, {name: scalar tensor})
    prepare(student, teacher, loader, distributed)   one-off setup on a fresh (not resumed) run
    before_step(student, epoch)      gradient surgery between backward and the optimizer step
    after_step(student, it)          per-iteration update after the optimizer step (DINO's EMA head)
    state_dict / load_state_dict     everything besides the student that a resume has to restore

s_out holds the student's 'score' (ID logits), 'global_feat' (pre-BNNeck CLS feature) and, for DINO, 'head'
(prototype similarities); t_out holds the teacher's 'global_feat' and 'logits'. Views are stacked along the
batch dim, view-major.
"""
import copy
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .losses import (DINOHead, EMACentering, dino_cross_entropy, entropy, koleo_loss, sinkhorn_knopp,
                     soft_kd_loss)
from .prototypes import kmeans_prototypes


class StudentWithHead(nn.Module):
    """ReID student plus the DINO projection head on its global feature, wrapped together so DDP syncs both."""
    def __init__(self, reid, head):
        super().__init__()
        self.reid = reid
        self.head = head

    def forward(self, img, label=None, cam_label=None, view_label=None):
        score, global_feat = self.reid(img, label, cam_label=cam_label, view_label=view_label)
        return score, global_feat, self.head(global_feat)


class KDMethod(nn.Module):
    student_views = 1
    teacher_views = 1

    def prepare(self, student, teacher, loader, distributed):
        pass

    def before_step(self, student, epoch):
        pass

    def after_step(self, student, it):
        pass


class NoKD(KDMethod):
    """Baseline: the student trains on the ReID losses alone; no teacher is built."""
    teacher_views = 0

    def forward(self, s_out, t_out, labels, epoch, with_stats=False):
        return s_out['score'].new_zeros(()), {}


class SoftKD(KDMethod):
    student_views = 1
    teacher_views = 1

    def __init__(self, cfg):
        super().__init__()
        self.temperature = cfg.KD.SOFT.TEMPERATURE
        self.weight = cfg.KD.SOFT.WEIGHT

    def forward(self, s_out, t_out, labels, epoch, with_stats=False):
        kl = soft_kd_loss(s_out['score'], t_out['logits'], self.temperature)
        stats = {'kd/soft_kl': kl.detach()}
        if with_stats:
            with torch.no_grad():
                stats['kd/top1_agreement'] = (s_out['score'].argmax(1) == t_out['logits'].argmax(1)).float().mean()
                stats['kd/teacher_acc'] = (t_out['logits'].argmax(1) == labels).float().mean()
        return self.weight * kl, stats


class DINOKD(KDMethod):
    """DINO-style distillation from a frozen ReID teacher.

    Targets: frozen teacher backbone -> teacher head -> Sinkhorn-Knopp (or EMA centering) at a sharp temperature.
    The student matches them through its own head. KD.DINO.TEACHER_HEAD picks the teacher head:

    'kmeans' (default): the prototypes are spherical k-means centroids of the teacher's L2-normalised features on
        the training set, computed once at the start and then frozen, shared by teacher and student. The teacher's
        scores are its features' cosines to the centroids, so its targets are meaningful from the first step; the
        student head (MLP -> 768-d) learns to project the student's features into the teacher's feature space.
        This mirrors DINOv2 distillation, where the frozen teacher keeps its trained head.
    'ema': DINO's scheme, teacher head = EMA of the student head (from the same random init). With a frozen,
        pretrained teacher backbone this gives near-uniform targets for a long time, because the head only becomes
        meaningful as fast as the student does (10-epoch Market test: DINO loss 8.44 -> 8.21 of log K = 8.32,
        ~1% student/teacher prototype agreement).

    Collapse guards: the teacher backbone is a trained, frozen network, so its features can't collapse; the teacher
    branch gets no gradient; Sinkhorn-Knopp forces every prototype to get equal mass per batch, so the targets
    can't concentrate on one prototype, and the sharp teacher temperature keeps them from going uniform; KoLeo
    spreads the student's features. 'ema' additionally freezes the prototypes for the first epoch(s).
    """

    def __init__(self, cfg, student_head, iters_per_epoch):
        super().__init__()
        c = cfg.KD.DINO
        self.weight = c.WEIGHT
        self.student_views = c.STUDENT_VIEWS
        self.teacher_views = cfg.INPUT.NUM_VIEWS
        self.student_temp = c.STUDENT_TEMP
        self.teacher_temp = c.TEACHER_TEMP
        self.warmup_teacher_temp = c.WARMUP_TEACHER_TEMP
        self.warmup_teacher_temp_epochs = c.WARMUP_TEACHER_TEMP_EPOCHS
        self.centering_mode = c.CENTERING
        self.sk_iters = c.SK_ITERS
        self.same_view_pairs = c.SAME_VIEW_PAIRS
        self.koleo_weight = c.KOLEO_WEIGHT
        self.koleo_exclude_same_id = c.KOLEO_EXCLUDE_SAME_ID
        self.base_momentum = c.HEAD_MOMENTUM
        self.freeze_last_layer_epochs = c.FREEZE_LAST_LAYER_EPOCHS
        self.total_iters = cfg.SOLVER.MAX_EPOCHS * iters_per_epoch
        self.teacher_head_mode = c.TEACHER_HEAD
        self.kmeans_iters = c.KMEANS_ITERS
        self.seed = cfg.SOLVER.SEED

        if self.teacher_head_mode == 'kmeans':
            # Filled by prepare() on a fresh run, restored from the checkpoint on a resumed one.
            self.register_buffer('prototypes', torch.zeros_like(student_head.prototypes.weight))
            student_head.prototypes.weight.requires_grad_(False)
        elif self.teacher_head_mode == 'ema':
            self.teacher_head = copy.deepcopy(student_head)
            for p in self.teacher_head.parameters():
                p.requires_grad_(False)
        else:
            raise ValueError('KD.DINO.TEACHER_HEAD must be kmeans or ema, got ' + self.teacher_head_mode)
        if self.centering_mode == 'centering':
            self.centering = EMACentering(c.OUT_DIM, c.CENTER_MOMENTUM)
        elif self.centering_mode != 'sinkhorn_knopp':
            raise ValueError('KD.DINO.CENTERING must be sinkhorn_knopp or centering, got ' + self.centering_mode)

    def current_teacher_temp(self, epoch):
        # Linear warmup over the first epochs (epochs are 1-based in the training loop), then constant.
        if epoch <= self.warmup_teacher_temp_epochs:
            frac = (epoch - 1) / max(self.warmup_teacher_temp_epochs, 1)
            return self.warmup_teacher_temp + frac * (self.teacher_temp - self.warmup_teacher_temp)
        return self.teacher_temp

    def current_momentum(self, it):
        # Cosine from base_momentum to 1 over training (DINO).
        return 1 - (1 - self.base_momentum) * (math.cos(math.pi * min(it, self.total_iters) / self.total_iters) + 1) / 2

    def prepare(self, student, teacher, loader, distributed):
        if self.teacher_head_mode != 'kmeans':
            return
        if not distributed or dist.get_rank() == 0:
            protos = kmeans_prototypes(teacher, loader, self.prototypes.shape[0], self.kmeans_iters, self.seed)
        else:
            protos = torch.empty_like(self.prototypes)
        if distributed:
            dist.broadcast(protos, src=0)
        self.prototypes.copy_(protos)
        student.head.prototypes.weight.data.copy_(protos)

    def teacher_logits(self, teacher_feat):
        if self.teacher_head_mode == 'kmeans':
            return F.linear(F.normalize(teacher_feat.float(), dim=-1), self.prototypes)
        return self.teacher_head(teacher_feat.float())

    def before_step(self, student, epoch):
        # DINO keeps the prototypes fixed early on: dropping their gradient makes the optimizer skip them.
        if self.teacher_head_mode == 'ema' and epoch <= self.freeze_last_layer_epochs:
            student.head.prototypes.weight.grad = None

    def forward(self, s_out, t_out, labels, epoch, with_stats=False):
        temp = self.current_teacher_temp(epoch)
        with torch.no_grad():
            t_logits = self.teacher_logits(t_out['global_feat'])
            if self.centering_mode == 'sinkhorn_knopp':
                t_probs = sinkhorn_knopp(t_logits, temp, self.sk_iters)
            else:
                t_probs = self.centering(t_logits, temp)
        s_logits = s_out['head']
        dino = dino_cross_entropy(s_logits.chunk(self.student_views), t_probs.chunk(self.teacher_views),
                                  self.student_temp, self.same_view_pairs)
        loss = self.weight * dino
        stats = {'kd/dino': dino.detach(), 'kd/teacher_temp': torch.tensor(temp)}

        if self.koleo_weight > 0:
            # Per student view, so the two views of one image aren't each other's nearest neighbour.
            koleo_labels = labels if self.koleo_exclude_same_id else None
            koleo = sum(koleo_loss(f, koleo_labels)
                        for f in s_out['global_feat'].chunk(self.student_views)) / self.student_views
            loss = loss + self.koleo_weight * koleo
            stats['kd/koleo'] = koleo.detach()

        if with_stats:
            # Collapse diagnostics. Per-sample entropy near log(K) = targets have gone uniform; batch-marginal
            # entropy near 0 = everything maps to one prototype; feat_std near 0 = the student's features collapsed.
            with torch.no_grad():
                s_probs = F.softmax(s_logits.float() / self.student_temp, dim=-1)
                feats = F.normalize(s_out['global_feat'].float(), dim=-1)
                stats.update({
                    'collapse/teacher_entropy': entropy(t_probs).mean(),
                    'collapse/teacher_marginal_entropy': entropy(t_probs.mean(0)),
                    'collapse/student_entropy': entropy(s_probs).mean(),
                    'collapse/student_marginal_entropy': entropy(s_probs.mean(0)),
                    'collapse/student_prototypes_used': torch.tensor(float(s_logits.argmax(1).unique().numel())),
                    'collapse/teacher_prototypes_used': torch.tensor(float(t_probs.argmax(1).unique().numel())),
                    'collapse/student_feat_std': feats.std(dim=0).mean(),
                    'kd/student_teacher_proto_agreement': (s_logits.argmax(1) == t_probs.argmax(1)[
                        :s_logits.shape[0]]).float().mean(),
                })
        return loss, stats

    @torch.no_grad()
    def after_step(self, student, it):
        if self.teacher_head_mode != 'ema':
            return
        m = self.current_momentum(it)
        for p_t, p_s in zip(self.teacher_head.parameters(), student.head.parameters()):
            p_t.mul_(m).add_(p_s.detach(), alpha=1 - m)


def build_kd(cfg, student, iters_per_epoch):
    """Returns (student, kd): DINO wraps the student together with its projection head."""
    method = cfg.KD.METHOD
    if method == 'none':
        return student, NoKD()
    if method == 'soft':
        return student, SoftKD(cfg)
    if method == 'dino':
        # With k-means prototypes the head projects into the teacher's feature space (ViT-B: 768-d, as the student).
        bottleneck = student.in_planes if cfg.KD.DINO.TEACHER_HEAD == 'kmeans' else cfg.KD.DINO.BOTTLENECK_DIM
        head = DINOHead(student.in_planes, cfg.KD.DINO.OUT_DIM, cfg.KD.DINO.HIDDEN_DIM, bottleneck)
        return StudentWithHead(student, head), DINOKD(cfg, head, iters_per_epoch)
    raise ValueError('KD.METHOD must be none, soft or dino, got ' + method)

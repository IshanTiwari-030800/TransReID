import hashlib
import logging

import torch
import torch.nn as nn

from config import default_cfg
from model import make_model


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


class FrozenTeacher(nn.Module):
    """A trained TransReID model, always in eval mode with no gradients.

    forward returns the same tensors the student exposes in training: the pre-BNNeck global feature (what the
    student's triplet loss and the DINO head see) and the ID logits from its classifier (what soft KD sees).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, img, cam_label=None, view_label=None):
        global_feat = self.model.base(img, cam_label=cam_label, view_label=view_label)
        logits = self.model.classifier(self.model.bottleneck(global_feat))
        return {'global_feat': global_feat, 'logits': logits}


def build_teacher(cfg, num_classes, camera_num, view_num):
    """Rebuild the teacher from the config it was trained with and load its trained weights (strictly)."""
    logger = logging.getLogger("transreid.train")
    tcfg = default_cfg()
    tcfg.merge_from_file(cfg.KD.TEACHER.CONFIG)
    tcfg.MODEL.PRETRAIN_CHOICE = 'none'  # the trained weights below replace everything; skip the ImageNet load
    tcfg.MODEL.GRAD_CHECKPOINT = False
    tcfg.freeze()

    if list(tcfg.INPUT.SIZE_TRAIN) != list(cfg.INPUT.SIZE_TRAIN):
        raise ValueError('teacher was trained at {} but the student trains at {}'.format(
            tcfg.INPUT.SIZE_TRAIN, cfg.INPUT.SIZE_TRAIN))
    if list(tcfg.INPUT.PIXEL_MEAN) != list(cfg.INPUT.PIXEL_MEAN) or \
            list(tcfg.INPUT.PIXEL_STD) != list(cfg.INPUT.PIXEL_STD):
        raise ValueError('teacher and student use different input normalisation')

    model = make_model(tcfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    state = torch.load(cfg.KD.TEACHER.WEIGHT, map_location='cpu')
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    n_cls = state['classifier.weight'].shape[0]
    if n_cls != num_classes:
        raise ValueError('teacher classifier has {} classes but the dataset has {}'.format(n_cls, num_classes))
    model.load_state_dict(state, strict=True)
    logger.info('Teacher: {} from {} (sha256 {})'.format(
        tcfg.MODEL.TRANSFORMER_TYPE, cfg.KD.TEACHER.WEIGHT, _sha256(cfg.KD.TEACHER.WEIGHT)))
    return FrozenTeacher(model), tcfg

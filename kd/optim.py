import torch


def make_kd_optimizer(cfg, model):
    """SGD or AdamW over the student (and DINO head). Biases, norm weights, position embeddings and the CLS token
    get no weight decay, as is standard when training a ViT from scratch (DeiT, DINO)."""
    no_decay_names = {'pos_embed', 'cls_token'}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.split('.')[-1] in no_decay_names:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{'params': decay, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
              {'params': no_decay, 'weight_decay': 0.0}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        return torch.optim.AdamW(groups, lr=cfg.SOLVER.BASE_LR)
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        return torch.optim.SGD(groups, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    raise ValueError('train_kd.py supports SOLVER.OPTIMIZER_NAME AdamW or SGD, got ' + cfg.SOLVER.OPTIMIZER_NAME)

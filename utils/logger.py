import logging
import os
import sys
import os.path as osp
def setup_logger(name, save_dir, if_train, resume=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Under DDP only rank 0 logs, so ranks don't interleave lines in stdout or train_log.txt.
    if int(os.environ.get("RANK", 0)) > 0:
        logger.propagate = False
        return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='a' if resume else 'w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
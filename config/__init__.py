# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .defaults import _C as cfg
from .defaults import _C as cfg_test

# Snapshot taken at import, before any yml is merged into cfg (which is modified in place).
_DEFAULTS = cfg.clone()


def default_cfg():
    """A fresh copy of the unmerged defaults, e.g. for building a second model (a KD teacher) from its own yml."""
    return _DEFAULTS.clone()

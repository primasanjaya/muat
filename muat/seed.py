"""Global random-seed control for reproducible muat runs.

The published pipeline previously left DataLoader shuffling and weight init
unseeded, so "deterministic" results were not actually reproducible. `set_seed`
pins every RNG muat touches (Python, NumPy, PyTorch CPU/CUDA) and switches cuDNN
into deterministic mode. Call it once, as early as possible, before building the
model or the dataloaders.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed, deterministic=True):
    """Seed all RNGs muat relies on.

    Args:
        seed (int): the seed value.
        deterministic (bool): if True, force cuDNN deterministic algorithms and
            disable the autotuner. Slightly slower but bit-reproducible on GPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("set random seed to {}{}".format(
        seed, " (deterministic)" if deterministic else ""))


def seed_worker(worker_id):
    """DataLoader ``worker_init_fn`` so each worker derives a stable seed from the
    torch base seed set by :func:`set_seed`. Pass this to ``DataLoader(..., worker_init_fn=seed_worker)``."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

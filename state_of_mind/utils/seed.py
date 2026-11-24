from __future__ import annotations

import random

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python inside a DataLoader worker from PyTorch's worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

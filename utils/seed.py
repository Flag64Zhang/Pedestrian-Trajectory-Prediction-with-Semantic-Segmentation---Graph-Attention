from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    # 尽量固定随机性，便于复现实验结果。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

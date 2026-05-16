from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    # 统一保存训练状态，便于恢复与评估。
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | None = None) -> Dict[str, Any]:
    # 加载检查点并映射到指定设备。
    return torch.load(path, map_location=map_location)

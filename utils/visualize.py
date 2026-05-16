from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_trajectories(
    scene_map: Optional[np.ndarray],
    obs: np.ndarray,
    fut: np.ndarray,
    preds: np.ndarray,
    save_path: str | Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualization.") from exc

    fig, ax = plt.subplots(figsize=(6, 6))
    if scene_map is not None:
        if scene_map.ndim == 3 and scene_map.shape[0] in {1, 3}:
            scene_map = np.transpose(scene_map, (1, 2, 0))
        ax.imshow(scene_map, alpha=0.7)

    ax.plot(obs[:, 0], obs[:, 1], "bo-", label="obs")
    ax.plot(fut[:, 0], fut[:, 1], "ro-", label="gt")

    # 支持单模态或多模态预测轨迹绘制。
    if preds.ndim == 2:
        ax.plot(preds[:, 0], preds[:, 1], "g--", label="pred")
    else:
        for k in range(preds.shape[0]):
            ax.plot(preds[k, :, 0], preds[k, :, 1], "g--", alpha=0.6)

    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

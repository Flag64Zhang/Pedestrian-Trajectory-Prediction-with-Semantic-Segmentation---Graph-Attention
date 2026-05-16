from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.paths import resolve_path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_trajectories(
    scene_map: Optional[np.ndarray],
    obs: np.ndarray,
    fut: np.ndarray,
    preds: np.ndarray,
    save_path: str | Path,
    ade: Optional[float] = None,
    fde: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    if scene_map is not None:
        if scene_map.ndim == 3 and scene_map.shape[0] in {1, 3}:
            scene_map = np.transpose(scene_map, (1, 2, 0))
        ax.imshow(scene_map, alpha=0.65, origin="upper")

    ax.plot(obs[:, 0], obs[:, 1], color="#1f77b4", linewidth=2.2, marker="o", markersize=5, label="观测")
    ax.plot(fut[:, 0], fut[:, 1], color="#d62728", linewidth=2.2, marker="o", markersize=5, label="真值")
    if preds.ndim == 2:
        ax.plot(preds[:, 0], preds[:, 1], color="#2ca02c", linewidth=2.4, linestyle="--", marker="s", markersize=4, label="预测")
        ax.plot(
            [obs[-1, 0], preds[0, 0]],
            [obs[-1, 1], preds[0, 1]],
            color="#2ca02c",
            linewidth=1.2,
            linestyle=":",
            alpha=0.7,
        )
    else:
        for k in range(preds.shape[0]):
            ax.plot(preds[k, :, 0], preds[k, :, 1], color="#2ca02c", linewidth=1.0, linestyle="--", alpha=0.25)
        ax.plot(preds[0, :, 0], preds[0, :, 1], color="#2ca02c", linewidth=2.4, linestyle="--", marker="s", markersize=4, label="预测")

    all_pts = np.concatenate([obs, fut, preds.reshape(-1, 2) if preds.ndim > 2 else preds], axis=0)
    margin = max(0.5, 0.08 * max(all_pts.max(axis=0) - all_pts.min(axis=0)))
    x0, x1 = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y0, y1 = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    title = "行人轨迹预测"
    if ade is not None and fde is not None:
        title += f"  |  ADE={ade:.3f}  FDE={fde:.3f}"
    ax.set_title(title, fontsize=11)

    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    if not save_path.is_file() or save_path.stat().st_size == 0:
        raise RuntimeError(f"图像写入失败: {save_path}")

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


# 轨迹级数据增强：随机旋转与缩放，仅作用于轨迹坐标。
class TrajectoryRotateScale:
    def __init__(
        self,
        prob: float = 0.5,
        rot_range: Tuple[float, float] = (-math.pi, math.pi),
        scale_range: Tuple[float, float] = (0.9, 1.1),
    ) -> None:
        self.prob = prob
        self.rot_range = rot_range
        self.scale_range = scale_range

    def __call__(self, obs, fut, neigh_traj, scene_map: Optional[np.ndarray]):
        if np.random.rand() > self.prob:
            return obs, fut, neigh_traj, scene_map

        angle = np.random.uniform(*self.rot_range)
        scale = np.random.uniform(*self.scale_range)
        rot = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32
        )

        def _apply(traj: np.ndarray) -> np.ndarray:
            return (traj @ rot.T) * scale

        obs = _apply(obs)
        fut = _apply(fut)
        neigh_traj = _apply(neigh_traj.reshape(-1, 2)).reshape(neigh_traj.shape)
        return obs, fut, neigh_traj, scene_map

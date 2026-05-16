from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    obs: np.ndarray
    fut: np.ndarray
    neighbors: List[np.ndarray]
    scene_id: str
    agent_id: int


# 读取场景语义图/特征图，统一为 numpy 数组。
def _load_scene_map(scene_dir: Optional[Path], scene_id: str, scene_ext: str) -> Optional[np.ndarray]:
    if scene_dir is None:
        return None
    path = scene_dir / f"{scene_id}{scene_ext}"
    if not path.exists():
        return None
    if scene_ext.lower() in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            return data["arr_0"]
        return data
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for image scene maps.") from exc
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _read_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=None)
    if data.ndim == 1:
        data = data[None, :]
    return data


# 基于滑动窗口生成样本，并在最后一帧计算邻居集合。
def _build_samples(
    data: np.ndarray,
    scene_id: str,
    obs_len: int,
    pred_len: int,
    skip: int,
    min_ped: int,
    neighbor_radius: float,
    max_neighbors: int,
) -> List[Sample]:
    seq_len = obs_len + pred_len
    frames = np.unique(data[:, 0]).astype(int)
    frames.sort()
    frame_data = [data[data[:, 0] == frame] for frame in frames]

    samples: List[Sample] = []
    num_frames = len(frames)
    for start_idx in range(0, num_frames - seq_len + 1, skip):
        window_data = np.concatenate(frame_data[start_idx : start_idx + seq_len], axis=0)
        ped_ids = np.unique(window_data[:, 1]).astype(int)

        ped_trajs: Dict[int, np.ndarray] = {}
        for pid in ped_ids:
            ped_seq = window_data[window_data[:, 1] == pid]
            if ped_seq.shape[0] != seq_len:
                continue
            ped_seq = ped_seq[np.argsort(ped_seq[:, 0])]
            ped_trajs[pid] = ped_seq[:, 2:4]

        if len(ped_trajs) < min_ped:
            continue

        for pid, traj in ped_trajs.items():
            obs = traj[:obs_len]
            fut = traj[obs_len:]
            neighbors: List[np.ndarray] = []
            for nid, ntraj in ped_trajs.items():
                if nid == pid:
                    continue
                dist = np.linalg.norm(ntraj[obs_len - 1] - obs[-1])
                if dist <= neighbor_radius:
                    neighbors.append(ntraj[:obs_len])
            if max_neighbors > 0:
                neighbors = neighbors[:max_neighbors]
            samples.append(
                Sample(obs=obs, fut=fut, neighbors=neighbors, scene_id=scene_id, agent_id=pid)
            )

    return samples


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        obs_len: int = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_ped: int = 1,
        neighbor_radius: float = 2.0,
        max_neighbors: int = 16,
        scene_dir: Optional[str | Path] = None,
        scene_ext: str = ".npy",
        transform: Optional[Callable] = None,
        normalize: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.min_ped = min_ped
        self.neighbor_radius = neighbor_radius
        self.max_neighbors = max_neighbors
        self.scene_dir = Path(scene_dir) if scene_dir else None
        self.scene_ext = scene_ext
        self.transform = transform
        self.normalize = normalize
        self.norm_eps = norm_eps

        self.samples: List[Sample] = []
        for path in sorted(self.split_dir.glob("*.txt")):
            data = _read_txt(path)
            self.samples.extend(
                _build_samples(
                    data,
                    scene_id=path.stem,
                    obs_len=obs_len,
                    pred_len=pred_len,
                    skip=skip,
                    min_ped=min_ped,
                    neighbor_radius=neighbor_radius,
                    max_neighbors=max_neighbors,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        obs = sample.obs.astype(np.float32)
        fut = sample.fut.astype(np.float32)
        scene = _load_scene_map(self.scene_dir, sample.scene_id, self.scene_ext)

        neighbors = sample.neighbors
        neigh_traj = np.zeros((self.max_neighbors, self.obs_len, 2), dtype=np.float32)
        neigh_mask = np.zeros((self.max_neighbors,), dtype=np.float32)
        for i, ntraj in enumerate(neighbors):
            if i >= self.max_neighbors:
                break
            neigh_traj[i] = ntraj.astype(np.float32)
            neigh_mask[i] = 1.0

        # 以最后观测点为原点做中心化，按最大绝对值缩放，稳定训练。
        center = obs[-1].copy()
        scale = 1.0
        if self.normalize:
            obs = obs - center
            fut = fut - center
            if self.max_neighbors > 0:
                neigh_traj = neigh_traj - center[None, None, :]
            max_val = np.max(np.abs(np.concatenate([obs, fut, neigh_traj.reshape(-1, 2)], axis=0)))
            scale = max(max_val, self.norm_eps)
            obs = obs / scale
            fut = fut / scale
            if self.max_neighbors > 0:
                neigh_traj = neigh_traj / scale

        if self.transform:
            obs, fut, neigh_traj, scene = self.transform(obs, fut, neigh_traj, scene)

        obs_t = torch.from_numpy(obs)
        fut_t = torch.from_numpy(fut)
        neigh_t = torch.from_numpy(neigh_traj)
        mask_t = torch.from_numpy(neigh_mask)
        scene_t = None
        if scene is not None:
            if scene.ndim == 2:
                scene = scene[:, :, None]
            scene_t = torch.from_numpy(scene.astype(np.float32)).permute(2, 0, 1)

        center_t = torch.from_numpy(center.astype(np.float32))
        scale_t = torch.tensor(scale, dtype=torch.float32)

        return obs_t, fut_t, scene_t, neigh_t, mask_t, idx, sample.agent_id, center_t, scale_t


# 统一批次维度，并保留反归一化所需的中心与尺度。
def collate_fn(batch):
    obs, fut, scene, neigh, mask, idx, agent, center, scale = zip(*batch)
    obs_t = torch.stack(obs, dim=0)
    fut_t = torch.stack(fut, dim=0)
    neigh_t = torch.stack(neigh, dim=0)
    mask_t = torch.stack(mask, dim=0)

    if all(s is None for s in scene):
        scene_t = None
    else:
        scene_t = torch.stack([s if s is not None else torch.zeros_like(scene[0]) for s in scene], dim=0)

    idx_t = torch.tensor(idx, dtype=torch.long)
    agent_t = torch.tensor(agent, dtype=torch.long)
    center_t = torch.stack(center, dim=0)
    scale_t = torch.stack(scale, dim=0)
    return obs_t, fut_t, scene_t, neigh_t, mask_t, idx_t, agent_t, center_t, scale_t

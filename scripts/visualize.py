from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset, collate_fn
from metrics.ade_fde import ade, fde
from models.csgat_net import CSGATNet
from utils.io import load_checkpoint
from utils.paths import load_config, resolve_config_paths, resolve_path, setup_project_env, to_rel_path
from utils.plotting import plot_trajectories


def _pick_best_pred(pred: np.ndarray, fut: np.ndarray) -> np.ndarray:
    if pred.ndim == 2:
        return pred
    dist = np.linalg.norm(pred - fut[None, :, :], axis=-1).mean(axis=1)
    return pred[int(dist.argmin())]


def _resolve_checkpoint(cfg: dict) -> Path:
    candidates = [
        cfg["eval"]["checkpoint"],
        "outputs/checkpoints/social/best.pt",
        "outputs/checkpoints/best.pt",
        "outputs/checkpoints/full/best.pt",
    ]
    seen = set()
    for ckpt in candidates:
        if not ckpt or ckpt in seen:
            continue
        seen.add(ckpt)
        path = resolve_path(ckpt)
        if path.exists():
            return path
    return resolve_path(cfg["eval"]["checkpoint"])


def _denorm_trajs(
    pred: np.ndarray,
    obs: np.ndarray,
    fut: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_np = obs * scale + center
    fut_np = fut * scale + center
    pred_np = pred * scale + center
    pred_np = _pick_best_pred(pred_np, fut_np)
    return obs_np, fut_np, pred_np


def _select_indices(
    model: CSGATNet,
    loader: DataLoader,
    device: torch.device,
    sample_k: int,
    num_images: int,
    start_index: int,
    select_mode: str,
    scan_limit: int,
    dataset_len: int,
) -> List[int]:
    if select_mode == "sequential":
        indices = list(range(start_index, start_index + num_images))
        if indices[-1] >= dataset_len:
            raise IndexError(
                f"连续取样超出范围: 需要索引 < {dataset_len}，请减小 num_images 或 start_index"
            )
        return indices

    scores: List[Tuple[float, int]] = []
    limit = min(scan_limit, dataset_len)
    with torch.no_grad():
        for i, (obs, fut, scene, neigh, mask, _, _, _, _) in enumerate(loader):
            if i >= limit:
                break
            obs = obs.to(device)
            fut = fut.to(device)
            neigh = neigh.to(device)
            mask = mask.to(device)
            scene_dev = scene.to(device) if scene is not None else None
            pred, _, _, _, _ = model(obs, None, scene_dev, neigh, mask, sample_k=sample_k)
            scores.append((ade(pred, fut).item(), i))

    scores.sort(key=lambda x: x[0])
    return [idx for _, idx in scores[:num_images]]


def _render_one(
    model: CSGATNet,
    batch,
    device: torch.device,
    sample_k: int,
    save_path: Path,
) -> Tuple[float, float]:
    obs, fut, scene, neigh, mask, _, _, center, scale = batch
    obs = obs.to(device)
    fut = fut.to(device)
    neigh = neigh.to(device)
    mask = mask.to(device)
    scene_dev = scene.to(device) if scene is not None else None

    pred, _, _, _, _ = model(obs, None, scene_dev, neigh, mask, sample_k=sample_k)
    ade_val = ade(pred, fut).item()
    fde_val = fde(pred, fut).item()

    pred_np = pred.squeeze(0).cpu().numpy()
    obs_np = obs.squeeze(0).cpu().numpy()
    fut_np = fut.squeeze(0).cpu().numpy()
    center_np = center.squeeze(0).cpu().numpy()
    scale_np = float(scale.squeeze(0).cpu().item())

    obs_np, fut_np, pred_np = _denorm_trajs(pred_np, obs_np, fut_np, center_np, scale_np)
    scene_np = scene.squeeze(0).cpu().numpy() if scene is not None else None

    plot_trajectories(scene_np, obs_np, fut_np, pred_np, save_path, ade=ade_val, fde=fde_val)
    return ade_val, fde_val


def main() -> None:
    setup_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visualize.yaml")
    parser.add_argument("--num", type=int, default=None, help="生成图像数量（默认 10）")
    parser.add_argument("--start", type=int, default=None, help="sequential 模式下的起始样本索引")
    args = parser.parse_args()

    cfg = resolve_config_paths(load_config(args.config))
    vis_cfg = cfg.get("visualize", {})
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    sample_k = int(vis_cfg.get("sample_k", cfg["eval"].get("sample_k", 40)))
    num_images = (
        args.num
        if args.num is not None
        else int(vis_cfg.get("num_images", 10))
    )
    num_images = max(1, num_images)
    start_index = args.start if args.start is not None else int(vis_cfg.get("start_index", 0))
    select_mode = vis_cfg.get("select_mode", "best_ade")
    scan_limit = int(vis_cfg.get("scan_limit", 300))

    dataset = TrajectoryDataset(
        split_dir=cfg["data"]["val_dir"],
        obs_len=cfg["data"]["obs_len"],
        pred_len=cfg["data"]["pred_len"],
        skip=cfg["data"]["skip"],
        min_ped=cfg["data"]["min_ped"],
        neighbor_radius=cfg["data"]["neighbor_radius"],
        max_neighbors=cfg["data"]["max_neighbors"],
        scene_dir=cfg["data"]["scene_dir"],
        scene_ext=cfg["data"]["scene_ext"],
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"验证集为空，请先执行菜单「1. 数据预处理」。数据目录: {cfg['data']['val_dir']}"
        )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = CSGATNet(
        obs_len=cfg["data"]["obs_len"],
        pred_len=cfg["data"]["pred_len"],
        hidden_dim=cfg["model"]["hidden_dim"],
        scene_dim=cfg["model"]["scene_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        use_scene=cfg["model"]["use_scene"],
        use_social=cfg["model"]["use_social"],
    ).to(device)

    ckpt_path = _resolve_checkpoint(cfg)
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"已加载模型: {to_rel_path(ckpt_path)}  (sample_k={sample_k})")
    else:
        print(f"警告: 未找到模型，将使用随机权重。请先训练: {to_rel_path(ckpt_path)}")
    model.eval()

    out_dir = resolve_path(vis_cfg.get("output_dir", "outputs/figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"计划生成 {num_images} 张图 (模式: {select_mode})")
    indices = _select_indices(
        model, loader, device, sample_k, num_images, start_index, select_mode, scan_limit, len(dataset)
    )
    if select_mode == "best_ade":
        print(
            f"已从验证集前 {min(scan_limit, len(dataset))} 条中"
            f"选取 ADE 最小的 {len(indices)} 个样本"
        )
    else:
        print(f"连续取样索引: {indices[0]} ~ {indices[-1]}")

    saved = 0
    ade_vals: List[float] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i not in indices:
                continue
            save_path = out_dir / f"sample_{i:03d}.png"
            ade_val, fde_val = _render_one(model, batch, device, sample_k, save_path)
            ade_vals.append(ade_val)
            print(f"  [{saved + 1}/{len(indices)}] {to_rel_path(save_path)}  ADE={ade_val:.4f}  FDE={fde_val:.4f}")
            saved += 1
            if saved >= len(indices):
                break

    if saved == 0:
        raise RuntimeError("未能生成任何可视化图像")
    print(f"\n共保存 {saved} 张图 -> {to_rel_path(out_dir)}/")
    print(f"平均 ADE={sum(ade_vals) / len(ade_vals):.4f}  平均 FDE 见各图标题")


if __name__ == "__main__":
    main()

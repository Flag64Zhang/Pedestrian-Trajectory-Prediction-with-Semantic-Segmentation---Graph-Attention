from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset, collate_fn
from models.csgat_net import CSGATNet
from utils.visualize import plot_trajectories


def _load_config(path: str | Path) -> Dict[str, Any]:
    # 读取配置，保持与训练参数一致。
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configs.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

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

    if Path(cfg["eval"]["checkpoint"]).exists():
        ckpt = torch.load(cfg["eval"]["checkpoint"], map_location=device)
        model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        for i, (obs, fut, scene, neigh, mask, _, _, center, scale) in enumerate(loader):
            if i != args.index:
                continue
            obs = obs.to(device)
            fut = fut.to(device)
            neigh = neigh.to(device)
            mask = mask.to(device)
            scene_dev = scene.to(device) if scene is not None else None

            pred, _, _, _, _ = model(obs, None, scene_dev, neigh, mask, sample_k=cfg["eval"]["sample_k"])
            pred_np = pred.squeeze(0).cpu().numpy()
            obs_np = obs.squeeze(0).cpu().numpy()
            fut_np = fut.squeeze(0).cpu().numpy()
            center_np = center.squeeze(0).cpu().numpy()
            scale_np = float(scale.squeeze(0).cpu().item())

            # 反归一化到原始坐标系，便于展示。
            obs_np = obs_np * scale_np + center_np
            fut_np = fut_np * scale_np + center_np
            pred_np = pred_np * scale_np + center_np
            scene_np = scene.squeeze(0).cpu().numpy() if scene is not None else None

            out_dir = Path(cfg["visualize"]["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_trajectories(scene_np, obs_np, fut_np, pred_np, out_dir / f"sample_{i}.png")
            break


if __name__ == "__main__":
    main()

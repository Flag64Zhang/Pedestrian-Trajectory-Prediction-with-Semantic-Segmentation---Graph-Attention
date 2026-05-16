from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset, collate_fn
from metrics.ade_fde import ade, fde
from models.csgat_net import CSGATNet
from utils.io import load_checkpoint


def _load_config(path: str | Path) -> Dict[str, Any]:
    # 与训练保持同一配置来源。
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configs.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    # 评估阶段保持与训练相同的设备策略。
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

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )

    model = CSGATNet(
        obs_len=cfg["data"]["obs_len"],
        pred_len=cfg["data"]["pred_len"],
        hidden_dim=cfg["model"]["hidden_dim"],
        scene_dim=cfg["model"]["scene_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        use_scene=cfg["model"]["use_scene"],
        use_social=cfg["model"]["use_social"],
    ).to(device)

    # 加载最佳模型并进行推理评估。
    ckpt = load_checkpoint(cfg["eval"]["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ade_list = []
    fde_list = []
    with torch.no_grad():
        for obs, fut, scene, neigh, mask, _, _, _, _ in loader:
            obs = obs.to(device)
            fut = fut.to(device)
            neigh = neigh.to(device)
            mask = mask.to(device)
            scene = scene.to(device) if scene is not None else None

            pred, _, _, _, _ = model(obs, None, scene, neigh, mask, sample_k=cfg["eval"]["sample_k"])
            ade_list.append(ade(pred, fut).item())
            fde_list.append(fde(pred, fut).item())

    ade_score = float(sum(ade_list) / max(1, len(ade_list)))
    fde_score = float(sum(fde_list) / max(1, len(fde_list)))
    print(f"ADE={ade_score:.4f} FDE={fde_score:.4f}")


if __name__ == "__main__":
    main()

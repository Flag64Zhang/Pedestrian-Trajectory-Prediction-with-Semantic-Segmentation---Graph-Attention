from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset, collate_fn
from losses.cvae_loss import total_cvae_loss
from metrics.ade_fde import ade, fde
from models.csgat_net import CSGATNet
from utils.io import ensure_dir, save_checkpoint
from utils.seed import set_seed


def _load_config(path: str | Path) -> Dict[str, Any]:
    # 配置统一从 YAML 读取，便于复现实验。
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configs.") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, device, beta):
    # 单轮训练：前向、损失、反传、更新。
    model.train()
    total_loss = 0.0
    for obs, fut, scene, neigh, mask, _, _, _, _ in loader:
        obs = obs.to(device)
        fut = fut.to(device)
        neigh = neigh.to(device)
        mask = mask.to(device)
        scene = scene.to(device) if scene is not None else None

        pred, prior_mu, prior_logvar, post_mu, post_logvar = model(obs, fut, scene, neigh, mask)
        loss = total_cvae_loss(pred, fut, prior_mu, prior_logvar, post_mu, post_logvar, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate(model, loader, device, sample_k: int):
    # 验证阶段使用 best-of-K 评估。
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

            pred, _, _, _, _ = model(obs, None, scene, neigh, mask, sample_k=sample_k)
            ade_list.append(ade(pred, fut).item())
            fde_list.append(fde(pred, fut).item())

    return float(sum(ade_list) / max(1, len(ade_list))), float(sum(fde_list) / max(1, len(fde_list)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    set_seed(cfg["train"]["seed"])

    # 优先使用配置指定设备，若无 GPU 则回退到 CPU。
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    train_ds = TrajectoryDataset(
        split_dir=cfg["data"]["train_dir"],
        obs_len=cfg["data"]["obs_len"],
        pred_len=cfg["data"]["pred_len"],
        skip=cfg["data"]["skip"],
        min_ped=cfg["data"]["min_ped"],
        neighbor_radius=cfg["data"]["neighbor_radius"],
        max_neighbors=cfg["data"]["max_neighbors"],
        scene_dir=cfg["data"]["scene_dir"],
        scene_ext=cfg["data"]["scene_ext"],
    )
    val_ds = TrajectoryDataset(
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

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    save_dir = ensure_dir(cfg["train"]["save_dir"])
    best_ade = float("inf")

    for epoch in range(cfg["train"]["epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, device, cfg["train"]["beta"])
        val_ade, val_fde = evaluate(model, val_loader, device, cfg["eval"]["sample_k"])

        if val_ade < best_ade:
            best_ade = val_ade
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_ade": val_ade,
                    "val_fde": val_fde,
                },
                save_dir / "best.pt",
            )
        print(f"epoch={epoch} loss={loss:.4f} val_ade={val_ade:.4f} val_fde={val_fde:.4f}")

    # 训练完成后保存 last 与 best，便于后续对比与复现。
    last_path = save_dir / "last.pt"
    save_checkpoint(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": cfg["train"]["epochs"] - 1,
        },
        last_path,
    )
    best_path = save_dir / "best.pt"
    if not best_path.exists():
        save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": cfg["train"]["epochs"] - 1,
            },
            best_path,
        )


if __name__ == "__main__":
    main()

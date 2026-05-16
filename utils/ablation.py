from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset, collate_fn
from metrics.ade_fde import ade, fde
from models.csgat_net import CSGATNet
from utils.io import load_checkpoint
from utils.paths import load_config, resolve_config_paths, resolve_path, to_rel_path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ABLATION_CONFIGS: Sequence[Tuple[str, str]] = (
    ("Baseline", "configs/ablation_baseline.yaml"),
    ("Scene-only", "configs/ablation_scene.yaml"),
    ("Social-only", "configs/ablation_social.yaml"),
    ("Full", "configs/ablation_full.yaml"),
)


def evaluate_model(cfg: Dict, device: torch.device) -> Tuple[float, float]:
    """在验证集上评估单个模型，返回 (ADE, FDE)。"""
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

    ckpt = load_checkpoint(cfg["eval"]["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ade_list: List[float] = []
    fde_list: List[float] = []
    sample_k = int(cfg["eval"]["sample_k"])
    with torch.no_grad():
        for obs, fut, scene, neigh, mask, _, _, _, _ in loader:
            obs = obs.to(device)
            fut = fut.to(device)
            neigh = neigh.to(device)
            mask = mask.to(device)
            scene_t = scene.to(device) if scene is not None else None
            pred, _, _, _, _ = model(obs, None, scene_t, neigh, mask, sample_k=sample_k)
            ade_list.append(ade(pred, fut).item())
            fde_list.append(fde(pred, fut).item())

    return float(sum(ade_list) / max(1, len(ade_list))), float(sum(fde_list) / max(1, len(fde_list)))


def run_ablation_eval(
    device: torch.device,
    configs: Sequence[Tuple[str, str]] = ABLATION_CONFIGS,
) -> List[Dict[str, object]]:
    """依次评估各消融配置，跳过缺失权重的模型。"""
    results: List[Dict[str, object]] = []
    for name, config_path in configs:
        cfg = resolve_config_paths(load_config(config_path))
        ckpt_path = resolve_path(cfg["eval"]["checkpoint"])
        if not ckpt_path.exists():
            print(f"  [跳过] {name}: 未找到 {to_rel_path(ckpt_path)}")
            continue
        print(f"  评估 {name} ...")
        ade_score, fde_score = evaluate_model(cfg, device)
        print(f"    ADE={ade_score:.4f}  FDE={fde_score:.4f}")
        results.append(
            {
                "name": name,
                "config": config_path,
                "checkpoint": to_rel_path(ckpt_path),
                "ade": ade_score,
                "fde": fde_score,
            }
        )
    return results


def plot_ablation_bar(
    results: Sequence[Dict[str, object]],
    save_path: str | Path,
    title: str = "消融实验 (ETH/UCY 验证集)",
) -> Path:
    """绘制 ADE/FDE 分组柱状图并保存。"""
    if not results:
        raise RuntimeError("无可用消融结果，请先训练各消融模型。")

    labels = [str(r["name"]) for r in results]
    ade_vals = [float(r["ade"]) for r in results]
    fde_vals = [float(r["fde"]) for r in results]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_ade = ax.bar(x - width / 2, ade_vals, width, label="ADE", color="#4C72B0", edgecolor="white")
    bars_fde = ax.bar(x + width / 2, fde_vals, width, label="FDE", color="#DD8452", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("误差 (越低越好)")
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    for bars in (bars_ade, bars_fde):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    save_path = resolve_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    return save_path


def _write_results_md(results: Sequence[Dict[str, object]], md_path: Path) -> None:
    lines = [
        "# Ablation Results",
        "",
        "| Model | ADE | FDE |",
        "| --- | --- | --- |",
    ]
    for r in results:
        lines.append(f"| {r['name']} | {float(r['ade']):.4f} | {float(r['fde']):.4f} |")
    lines.extend(["", "Notes:", "- ADE lower is better; FDE lower is better.", ""])
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_ablation_and_plot(
    device: Optional[torch.device] = None,
    output_path: str | Path = "outputs/figures/ablation_bar.png",
    results_md: str | Path = "references/ablation_results.md",
) -> Optional[Path]:
    """运行消融评估并生成柱状图，返回图像路径。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n>>> 消融实验评估")
    results = run_ablation_eval(device)
    if not results:
        print("未能生成消融图：无可用检查点。请先训练 baseline/scene/social/full 模型。")
        return None

    out = plot_ablation_bar(results, output_path)
    _write_results_md(results, resolve_path(results_md))
    print(f"已保存消融柱状图: {to_rel_path(out)}")
    print(f"已更新结果表: {to_rel_path(results_md)}")
    return out

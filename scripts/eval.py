from __future__ import annotations

import argparse

import torch

from utils.ablation import evaluate_model, run_ablation_and_plot
from utils.paths import load_config, resolve_config_paths, setup_project_env


def main() -> None:
    setup_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--plot-ablation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="评估结束后生成消融实验柱状图 ablation_bar.png",
    )
    args = parser.parse_args()

    cfg = resolve_config_paths(load_config(args.config))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    print(">>> 当前模型评估")
    ade_score, fde_score = evaluate_model(cfg, device)
    print(f"ADE={ade_score:.4f} FDE={fde_score:.4f}")

    if args.plot_ablation:
        run_ablation_and_plot(device=device)


if __name__ == "__main__":
    main()

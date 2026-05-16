from __future__ import annotations

import torch


def _pairwise_displacement(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target, dim=-1)


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 4:
        # (B, K, T, 2) 使用 best-of-K 评估。
        disp = _pairwise_displacement(pred, target[:, None, :, :])
        ade_k = disp.mean(dim=-1)
        return ade_k.min(dim=1).values.mean()
    disp = _pairwise_displacement(pred, target)
    return disp.mean()


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 4:
    # (B, K, T, 2) 使用 best-of-K 评估。
        disp = _pairwise_displacement(pred[:, :, -1], target[:, None, -1])
        return disp.min(dim=1).values.mean()
    disp = _pairwise_displacement(pred[:, -1], target[:, -1])
    return disp.mean()

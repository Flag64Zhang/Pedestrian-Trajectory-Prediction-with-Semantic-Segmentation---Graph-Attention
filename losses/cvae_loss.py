from __future__ import annotations

import torch


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def kl_divergence(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return torch.mean(torch.sum(kl, dim=-1))

# 重构损失 + KL 散度。
def total_cvae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    post_mu: torch.Tensor,
    post_logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    recon = reconstruction_loss(pred, target)
    kl = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar)
    return recon + beta * kl

from __future__ import annotations

import torch
from torch import nn


class CVAEHead(nn.Module):
    def __init__(self, cond_dim: int, fut_dim: int, latent_dim: int = 16, hidden_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # 后验：结合条件与未来轨迹；先验：仅条件。
        self.posterior = nn.Sequential(
            nn.Linear(cond_dim + fut_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.prior = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    @staticmethod
    def _split(mu_logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    # 重参数采样。
    def sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, cond_feat: torch.Tensor, fut_feat: torch.Tensor | None = None):
        prior_mu, prior_logvar = self._split(self.prior(cond_feat))
        if fut_feat is None:
            z = self.sample(prior_mu, prior_logvar)
            return z, prior_mu, prior_logvar, None, None

        post_mu, post_logvar = self._split(self.posterior(torch.cat([cond_feat, fut_feat], dim=-1)))
        z = self.sample(post_mu, post_logvar)
        return z, prior_mu, prior_logvar, post_mu, post_logvar

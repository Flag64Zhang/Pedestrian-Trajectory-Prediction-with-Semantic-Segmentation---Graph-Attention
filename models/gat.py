from __future__ import annotations

import torch
from torch import nn


class SocialGAT(nn.Module):
    def __init__(self, hidden_dim: int = 64, attn_dim: int = 32) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    # 基于掩码的简化注意力聚合邻居特征。
    def forward(self, agent_feat: torch.Tensor, neigh_feats: torch.Tensor, neigh_mask: torch.Tensor) -> torch.Tensor:
        if neigh_feats.numel() == 0:
            return torch.zeros_like(agent_feat)

        bsz, num_neigh, _ = neigh_feats.shape
        agent_exp = agent_feat[:, None, :].expand(-1, num_neigh, -1)
        x = torch.cat([agent_exp, neigh_feats], dim=-1)
        attn = self.score(torch.tanh(self.fc(x))).squeeze(-1)

        mask = neigh_mask.float()
        attn = attn.masked_fill(mask <= 0, -1e9)
        weights = torch.softmax(attn, dim=-1)
        weights = weights * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        g_feat = torch.sum(weights[:, :, None] * neigh_feats, dim=1)
        return g_feat

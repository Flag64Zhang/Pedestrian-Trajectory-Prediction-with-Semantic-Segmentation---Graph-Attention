from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .cvae_head import CVAEHead
from .gat import SocialGAT
from .lstm_encoder import LSTMEncoder
from .resnet_seg import SceneEncoder


class CSGATNet(nn.Module):
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_dim: int = 64,
        scene_dim: int = 128,
        latent_dim: int = 16,
        use_scene: bool = True,
        use_social: bool = True,
    ) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.scene_dim = scene_dim
        self.latent_dim = latent_dim
        self.use_scene = use_scene
        self.use_social = use_social

        self.traj_encoder = LSTMEncoder(input_dim=2, hidden_dim=hidden_dim)
        self.neigh_encoder = LSTMEncoder(input_dim=2, hidden_dim=hidden_dim)
        self.scene_encoder = SceneEncoder(in_channels=3, out_dim=scene_dim) if use_scene else None
        self.social_gat = SocialGAT(hidden_dim=hidden_dim) if use_social else None

        # 按配置拼接场景与社交特征，再统一映射到条件向量。
        fuse_dim = hidden_dim
        if use_scene:
            fuse_dim += scene_dim
        if use_social:
            fuse_dim += hidden_dim

        self.fuse = nn.Sequential(nn.Linear(fuse_dim, hidden_dim), nn.ReLU())
        # 未来轨迹编码仅用于训练阶段的后验分布。
        self.fut_encoder = nn.Sequential(nn.Flatten(), nn.Linear(pred_len * 2, hidden_dim), nn.ReLU())
        self.cvae = CVAEHead(cond_dim=hidden_dim, fut_dim=hidden_dim, latent_dim=latent_dim)

        self.dec_init = nn.Linear(hidden_dim + latent_dim, hidden_dim * 2)
        self.dec_input = nn.Linear(2, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out_fc = nn.Linear(hidden_dim, 2)

    # 将邻居轨迹编码成特征序列，供注意力模块聚合。
    def _encode_neighbors(self, neigh_traj: torch.Tensor, neigh_mask: torch.Tensor) -> torch.Tensor:
        bsz, num_neigh, tlen, _ = neigh_traj.shape
        if num_neigh == 0:
            return torch.zeros((bsz, 0, self.hidden_dim), device=neigh_traj.device)

        flat = neigh_traj.view(bsz * num_neigh, tlen, 2)
        feats = self.neigh_encoder(flat).view(bsz, num_neigh, -1)
        return feats

    # 自回归解码，逐步预测未来位移并累加到坐标上。
    def _decode(self, last_obs: torch.Tensor, cond_feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        init = self.dec_init(torch.cat([cond_feat, z], dim=-1))
        h0, c0 = init.chunk(2, dim=-1)
        h = h0.unsqueeze(0).contiguous()
        c = c0.unsqueeze(0).contiguous()

        preds = []
        prev = last_obs
        for _ in range(self.pred_len):
            inp = self.dec_input(prev).unsqueeze(1)
            out, (h, c) = self.decoder(inp, (h, c))
            delta = self.out_fc(out.squeeze(1))
            pred = prev + delta
            preds.append(pred)
            prev = pred

        return torch.stack(preds, dim=1)

    def forward(
        self,
        obs_traj: torch.Tensor,
        fut_traj: Optional[torch.Tensor],
        scene_map: Optional[torch.Tensor],
        neigh_traj: torch.Tensor,
        neigh_mask: torch.Tensor,
        sample_k: int = 1,
    ):
        # 轨迹、场景、社交三路特征融合为条件向量。
        agent_feat = self.traj_encoder(obs_traj)
        feats = [agent_feat]

        if self.use_scene:
            if scene_map is None:
                scene_feat = torch.zeros((obs_traj.size(0), self.scene_dim), device=obs_traj.device)
            else:
                scene_feat = self.scene_encoder(scene_map)
            feats.append(scene_feat)

        if self.use_social:
            neigh_feats = self._encode_neighbors(neigh_traj, neigh_mask)
            social_feat = self.social_gat(agent_feat, neigh_feats, neigh_mask)
            feats.append(social_feat)

        cond_feat = self.fuse(torch.cat(feats, dim=-1))

        # 训练：使用后验分布；推理：使用先验分布并采样多模态。
        if fut_traj is not None:
            fut_feat = self.fut_encoder(fut_traj)
            z, prior_mu, prior_logvar, post_mu, post_logvar = self.cvae(cond_feat, fut_feat)
            pred = self._decode(obs_traj[:, -1], cond_feat, z)
            return pred, prior_mu, prior_logvar, post_mu, post_logvar

        preds = []
        prior_mu = None
        prior_logvar = None
        for _ in range(sample_k):
            z, prior_mu, prior_logvar, _, _ = self.cvae(cond_feat, None)
            preds.append(self._decode(obs_traj[:, -1], cond_feat, z))
        pred = torch.stack(preds, dim=1)
        return pred, prior_mu, prior_logvar, None, None

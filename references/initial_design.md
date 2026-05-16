# Initial Code Design (CSGAT-Net Simplified)

## 1. Goals and scope
- Implement a simplified trajectory prediction model combining: trajectory encoding, scene semantic features, social interaction, and a CVAE head for multi-modal outputs.
- Use ETH/UCY as the baseline dataset with optional semantic maps.
- Provide a clean, modular PyTorch codebase for coursework-level experiments and visualization.

## 2. Data assumptions
- Each sample contains:
  - history: observed trajectory positions (T_obs, 2)
  - future: target trajectory positions (T_pred, 2)
  - scene: semantic map image or precomputed scene feature map
  - neighbors: nearby agents history trajectories (N, T_obs, 2)
- Coordinate system is consistent between trajectory coordinates and scene map.

## 3. Proposed directory layout
- data/
  - dataset.py
  - transforms.py
- models/
  - lstm_encoder.py
  - resnet_seg.py
  - gat.py
  - cvae_head.py
  - csgat_net.py
- losses/
  - cvae_loss.py
- metrics/
  - ade_fde.py
- utils/
  - seed.py
  - visualize.py
  - io.py
- configs/
  - default.yaml
- scripts/
  - train.py
  - eval.py
  - visualize.py
- experiments/
  - baseline_lstm/
  - lstm_scene/
  - lstm_social/
  - lstm_scene_social/

## 4. Module design

### 4.1 data/dataset.py
- Class: TrajectoryDataset
- __getitem__ returns:
  - obs_traj: (T_obs, 2)
  - fut_traj: (T_pred, 2)
  - scene_map: (C, H, W) or None
  - neigh_traj: (N, T_obs, 2) padded to max_neighbors
  - neigh_mask: (N,) valid mask
  - seq_id, agent_id for tracking
- Responsibilities:
  - Load ETH/UCY split files
  - Build neighbor sets by spatial radius at last observed timestep
  - Read semantic map or load precomputed scene features

### 4.2 models/lstm_encoder.py
- Class: LSTMEncoder
- Inputs: obs_traj (B, T_obs, 2)
- Output: h_obs (B, H)
- Architecture:
  - Linear(2 -> H)
  - LSTM(H -> H)
  - Use last hidden state as trajectory embedding

### 4.3 models/resnet_seg.py
- Class: SceneEncoder
- Inputs: scene_map (B, C, H, W)
- Output: s_feat (B, S)
- Architecture:
  - ResNet backbone with global average pooling
  - Optional 1x1 conv to reduce channels
  - Linear to S for fusion
- Note: can be replaced by a lightweight CNN if compute is limited

### 4.4 models/gat.py
- Class: SocialGAT
- Inputs:
  - agent_feat: (B, H)
  - neigh_feats: (B, N, H)
  - neigh_mask: (B, N)
- Output: g_feat (B, H)
- Simplified attention:
  - Compute attention weights from concat(agent, neighbor)
  - Mask invalid neighbors
  - Weighted sum of neighbor features

### 4.5 models/cvae_head.py
- Class: CVAEHead
- Inputs:
  - cond_feat: (B, F) fused feature
  - fut_traj (B, T_pred, 2) during training
- Outputs:
  - z_mu, z_logvar
  - z sampled during training and inference
- Notes:
  - Posterior network uses cond_feat + future encoding
  - Prior network uses cond_feat only

### 4.6 models/csgat_net.py
- Class: CSGATNet
- Components:
  - LSTMEncoder for agent
  - LSTMEncoder for neighbors (shared weights)
  - SceneEncoder for semantic map
  - SocialGAT for interaction
  - Fusion MLP
  - CVAEHead
  - Decoder LSTM for future trajectory
- Forward (train):
  1) Encode agent and neighbors
  2) Encode scene
  3) SocialGAT for interaction feature
  4) Fuse (concat + MLP)
  5) CVAEHead (posterior)
  6) Decode future with z + fused feature
- Forward (eval):
  - Use prior z from cond_feat
  - Sample K trajectories for multi-modal output

## 5. Losses and metrics

### 5.1 losses/cvae_loss.py
- Reconstruction loss: L2/MSE on predicted future
- KL divergence: posterior vs prior
- Total: L = recon + beta * KL

### 5.2 metrics/ade_fde.py
- ADE: average displacement error over all predicted steps
- FDE: final displacement error at last step

## 6. Training and evaluation flow

### scripts/train.py
- Load config
- Build dataset/dataloader
- Initialize model, optimizer, scheduler
- Train epochs:
  - forward
  - compute loss
  - backprop
  - log metrics
- Save best checkpoint by ADE

### scripts/eval.py
- Load checkpoint
- Run inference with K samples
- Report ADE/FDE for each experiment

### scripts/visualize.py
- Render scene map + observed, future, predicted trajectories
- Save figures for report

## 7. Small architecture modifications (and reasons)
1) Add configs/ (default.yaml) to make experiments reproducible and easy to change.
2) Split CVAE logic into models/cvae_head.py to keep csgat_net.py focused on wiring.
3) Add metrics/ade_fde.py so evaluation is consistent across experiments.
4) Add scripts/ and experiments/ directories for repeatable training/evaluation runs.
5) Add data/transforms.py to isolate optional augmentations (rotation, scaling) without changing dataset core.

## 8. Baseline experiment set
- Baseline: LSTM only (no scene, no social)
- Model A: LSTM + scene
- Model B: LSTM + social
- Full: LSTM + scene + social

## 9. Next steps for implementation
- Define the ETH/UCY loader and neighbor sampling logic first.
- Implement LSTMEncoder and simple decoder to get the baseline running.
- Add SceneEncoder and SocialGAT, then integrate CVAE.
- Build visualization to validate qualitative behavior.

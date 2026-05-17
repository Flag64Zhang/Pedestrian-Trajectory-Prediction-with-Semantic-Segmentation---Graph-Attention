# 概述

本项目实现了一个用于**行人轨迹预测**的深度学习模型，它将三个互补的信息流——观测运动历史、行人间的社会交互以及场景语义上下文——融合到一个名为 **CSGAT-Net** 的统一框架中。系统的核心是：使用 LSTM 编码观测的 8 帧轨迹，通过基于图注意力机制的邻居行人交互模块和基于 ResNet 的场景语义分割编码器对场景图像进行特征增强表示，随后通过条件变分自编码器 (CVAE) 解码出多模态的未来轨迹。整个流水线十分轻量级 (hidden_dim=64, latent_dim=16)，可通过 YAML 文件进行配置，并在标准的 ETH/UCY 基准数据集上使用 ADE/FDE 指标结合 best-of-K 采样进行评估。无论你是首次探索轨迹预测，还是在寻找一个简洁、模块化的代码库进行扩展，本仓库都提供了一个结构良好的基础。

来源: [README.md](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/README.md#L1-L94), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L1-L127), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L1-L45)

------

## 为什么本项目很重要

行人轨迹预测处于自动驾驶、机器人导航和智能监控的交汇点。其核心挑战在于人类运动本质上是**多模态**的——在走廊交叉口的人可能会左转或右转——并且同时受到**社会力**（避免碰撞、结伴而行）和**物理约束**（人行道、障碍物、墙壁）的影响。传统的纯 LSTM 方法虽然能捕捉时序动态，却忽略了这些更丰富的信号。本项目证明了，将基于图注意力的社会建模和基于 ResNet 的场景理解集成到 CVAE 框架中，可以在保持架构足够简单（单张 GPU 不足 20 个 epoch 即可完成训练）的同时，取得具有竞争力的 FDE 性能。模块化设计——通过 `use_scene` 和 `use_social` 标志控制——也使得该代码库成为消融实验的理想测试平台。

来源: [README.md](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/README.md#L7-L12), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L25-L30), [ablation_results.md](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/references/ablation_results.md#L1-L14)

------

## 架构一览

下图展示了完整的 CSGAT-Net 前向传播过程。三个并行的编码器分支分别为目标智能体的轨迹、其社会邻域和周围场景生成特征向量。这些特征被拼接并投影为一个统一的条件向量，CVAE 头部利用该向量与潜变量 *z* 一起，初始化一个自回归 LSTM 解码器，从而预测 12 个未来的位移步长。







模型构造函数中的 `use_scene` 和 `use_social` 布尔标志控制着哪些分支处于激活状态。当某个分支被禁用时，其特征维度会从融合层的输入大小中排除——这不是零掩码技巧，而是真正的架构变更，这就是为什么消融配置会产生真正不同的模型。



来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L32-L52), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L88-L127)

------

## 模型组件概览

CSGAT-Net 中的每个组件都有单一且明确的职责。下表将每个模块映射到其源文件、输入/输出形状，以及你可以找到其深入文档的目录页面。

| 组件              | 源文件                                                       | 输入形状                                           | 输出形状            | 深入了解                                                     |
| ----------------- | ------------------------------------------------------------ | -------------------------------------------------- | ------------------- | ------------------------------------------------------------ |
| **LSTM 编码器**   | [lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py) | `(B, T, 2)`                                        | `(B, 64)`           | [轨迹 LSTM 编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/4-trajectory-lstm-encoder) |
| **SocialGAT**     | [gat.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/gat.py) | 智能体 `(B, 64)`，邻居 `(B, N, 64)`，掩码 `(B, N)` | `(B, 64)`           | [社会图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network) |
| **SceneEncoder**  | [resnet_seg.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/resnet_seg.py) | `(B, 3, H, W)`                                     | `(B, 128)`          | [ResNet 场景语义编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/6-resnet-scene-semantic-encoder) |
| **CVAE Head**     | [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py) | 条件 `(B, 64)`，未来 `(B, 64)`                     | `z (B, 16)`，μ/σ 对 | [CVAE 多模态轨迹头部](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/7-cvae-multimodal-trajectory-head) |
| **融合 + 解码器** | [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py) | 融合 `(B, fuse_dim)`                               | 预测 `(B, 12, 2)`   | [特征融合与自回归解码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/8-feature-fusion-and-autoregressive-decoder) |

来源: [lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py#L1-L24), [gat.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/gat.py#L1-L31), [resnet_seg.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/resnet_seg.py#L1-L45), [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L1-L43), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L32-L52)

------

## 消融实验结果

本项目提供了四个 YAML 配置文件，用于系统地切换场景和社会模块。在 ETH/UCY 验证数据上使用 best-of-20 采样评估的结果揭示了一个明显的规律：**社会交互建模对 ADE 的贡献最大**，而**完整模型取得了最佳的 FDE**，这表明场景上下文主要有助于减少终点误差。

| 配置                   | `use_scene` | `use_social` | ADE ↓      | FDE ↓      |
| ---------------------- | ----------- | ------------ | ---------- | ---------- |
| 基线 (仅 LSTM)         | `false`     | `false`      | 0.0132     | 0.0247     |
| 仅场景                 | `true`      | `false`      | 0.0196     | 0.0325     |
| 仅社会                 | `false`     | `true`       | **0.0118** | 0.0210     |
| **完整 (场景 + 社会)** | `true`      | `true`       | 0.0137     | **0.0184** |





仅场景模型在两项指标上的表现均逊于基线——这并不是 Bug。当场景图缺乏细粒度的可行走性信息时，单独的场景特征可能会引入噪声；只有与将预测建立在观测到的行人行为之上的社会特征结合时，它们才会变得有益。



来源: [ablation_results.md](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/references/ablation_results.md#L1-L14), [ablation_baseline.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_baseline.yaml#L17-L20), [ablation_scene.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_scene.yaml#L17-L20), [ablation_social.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_social.yaml#L17-L20), [ablation_full.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_full.yaml#L17-L20)

------

## 项目结构

代码库遵循清晰的关注点分离原则：数据处理、模型定义、训练逻辑和工具函数各自位于独立的顶层目录中。下图提供了每个模块及其关系的可视化映射。



```
Pedestrian-Trajectory-Prediction/
│
├── configs/              ← YAML 配置 (默认 + 4 个消融变体)
├── data/
│   ├── dataset.py        ← TrajectoryDataset, Sample, collate_fn
│   ├── transforms.py     ← TrajectoryRotateScale 数据增强
│   ├── crowds/           ← 原始 UCY 数据集文件
│   ├── ewap_dataset/     ← 原始 ETH 数据集文件 (seq_eth, seq_hotel)
│   └── processed/        ← 预处理的训练/验证文本文件
│
├── models/
│   ├── csgat_net.py      ← 顶层 CSGATNet 模块 (融合 + 解码器)
│   ├── lstm_encoder.py   ← 智能体与邻居轨迹共享的 LSTM 编码器
│   ├── gat.py            ← SocialGAT (对邻居的掩码注意力)
│   ├── resnet_seg.py     ← SceneEncoder (ResNet-18 主干 + 投影)
│   └── cvae_head.py      ← CVAEHead (先验、后验、重参数化)
│
├── losses/
│   └── cvae_loss.py      ← 重建 + KL 散度损失
├── metrics/
│   └── ade_fde.py        ← 带有 best-of-K 选择的 ADE / FDE
│
├── scripts/
│   ├── preprocess_eth_ucy.py  ← 解析原始 ETH/UCY → 标准化文本
│   ├── train.py               ← 带有检查点的训练循环
│   ├── eval.py                ← 评估脚本 (报告 ADE/FDE)
│   └── visualize.py           ← 生成轨迹预测图
│
├── utils/
│   ├── io.py             ← 检查点保存/加载，目录辅助工具
│   ├── seed.py           ← 确定性种子设置器
│   └── visualize.py      ← 用于 obs/gt/pred 叠加的 Matplotlib 绘图
│
├── experiments/          ← 实验日志 (基线、场景、社会、完整)
└── outputs/
    ├── checkpoints/      ← 保存的模型权重 (每个变体的 best.pt)
    └── figures/          ← 生成的可视化 PNG 图片
```

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L1-L6), [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L1-L16), [cvae_loss.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/losses/cvae_loss.py#L1-L34), [ade_fde.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/metrics/ade_fde.py#L1-L27), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L1-L15)

------

## 数据流管线

了解原始轨迹数据如何转化为训练好的预测模型，对于驾驭本代码库至关重要。以下流程图描绘了从数据集下载到可视化输出的完整管线。



来源: [preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L93-L185), [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L85-L155), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L33-L62), [eval.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/eval.py#L56-L91), [visualize.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/visualize.py#L46-L93)

------

## 核心配置参数

所有运行时行为均由单个 YAML 配置文件控制，该文件由 `train.py` 和 `eval.py` 共同加载。下表总结了 [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml) 中最具影响力的参数。

| 部分      | 参数              | 默认值 | 描述                              |
| --------- | ----------------- | ------ | --------------------------------- |
| **data**  | `obs_len`         | 8      | 观测时间步数                      |
|           | `pred_len`        | 12     | 预测的未来时间步数                |
|           | `neighbor_radius` | 2.0    | 邻居选择的欧氏距离阈值 (米)       |
|           | `max_neighbors`   | 16     | 每个智能体的最大邻居数 (用零填充) |
| **model** | `hidden_dim`      | 64     | LSTM 隐藏层大小和融合输出维度     |
|           | `scene_dim`       | 128    | 场景编码器输出维度                |
|           | `latent_dim`      | 16     | CVAE 潜变量维度                   |
|           | `use_scene`       | true   | 启用/禁用场景编码器分支           |
|           | `use_social`      | true   | 启用/禁用社会 GAT 分支            |
| **train** | `batch_size`      | 32     | 小批量大小                        |
|           | `epochs`          | 20     | 总训练轮数                        |
|           | `lr`              | 0.001  | Adam 学习率                       |
|           | `beta`            | 1.0    | KL 散度权重 (β-VAE 调度)          |
| **eval**  | `sample_k`        | 20     | 用于 best-of-K 评估的 CVAE 采样数 |

来源: [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L1-L45)

------

## 预测样本

本项目生成的可视化叠加图在场景图像上方渲染了观测轨迹 (蓝色)、真实未来轨迹 (红色) 和预测未来轨迹 (绿色)。以下是验证集的一个输出示例。

![Sample prediction overlay](z-read/sample_0.png)

来源: [visualize.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/visualize.py#L46-L93), [plot function](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/utils/visualize.py#L1-L43)

------

## 推荐阅读路径

本文档按渐进式学习路径组织。如果你是本项目的新手，请先阅读 **入门** 页面以设置环境并运行训练。然后进入 **深入理解** 部分，在攻克训练和评估管线之前，单独理解每个组件。

1. **入门** — 设置你的环境并运行基线模型
   - → [快速入门](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/2-quick-start)
2. **架构概览** — 在深入组件之前了解全局
   - → [架构概览](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/3-architecture-overview)
3. **模型组件** — 按数据流顺序阅读：编码器 → 注意力 → 融合 → 解码器
   - → [轨迹 LSTM 编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/4-trajectory-lstm-encoder)
   - → [社会图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network)
   - → [ResNet 场景语义编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/6-resnet-scene-semantic-encoder)
   - → [CVAE 多模态轨迹头部](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/7-cvae-multimodal-trajectory-head)
   - → [特征融合与自回归解码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/8-feature-fusion-and-autoregressive-decoder)
4. **数据管线** — 了解原始 ETH/UCY 文件如何转化为模型可用的张量
   - → [ETH/UCY 数据集预处理](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/9-eth-ucy-dataset-preprocessing)
   - → [轨迹数据集与邻居构建](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/10-trajectory-dataset-and-neighbor-construction)
   - → [数据增强变换](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/11-data-augmentation-transforms)
5. **训练与评估** — 损失函数、评估指标和消融设计
   - → [训练循环与检查点](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/12-training-loop-and-checkpointing)
   - → [CVAE 损失函数](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/13-cvae-loss-function)
   - → [带有 Best-of-K 的 ADE/FDE 指标](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/14-ade-fde-metrics-with-best-of-k)
   - → [配置与消融设置](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/15-configuration-and-ablation-setup)
6. **可视化** — 解释你的模型预测
   - → [预测可视化管线](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/16-prediction-visualization-pipeline)

# 快速开始

在 10 分钟内端到端运行 CSGAT-Net 行人轨迹预测流水线。本指南将带你走过环境配置、数据准备、训练、评估和可视化——涵盖从原始 ETH/UCY 数据到在屏幕上渲染出预测轨迹的完整工作流。以下所有命令均假设你位于项目根目录。

## 前提条件

本项目依赖于一个极简的 Python 技术栈。请首先创建一个干净的虚拟环境并安装所需的依赖包：



```
python -m venv venvsource venv/bin/activate      # Linux / macOS# venv\Scripts\activate       # Windows # 安装核心依赖pip install torch torchvision   # PyTorch >= 1.7pip install numpy pandas matplotlib pyyaml
```

下表列出了每个依赖项在流水线中的具体作用，让你确切了解*为什么*需要每一个包：

| 包                | 作用                         | 使用位置                              |
| ----------------- | ---------------------------- | ------------------------------------- |
| **PyTorch** ≥ 1.7 | 张量运算、自动求导、GPU 加速 | `models/`, `scripts/`                 |
| **TorchVision**   | 场景编码器的 ResNet 主干网络 | `models/resnet_seg.py`                |
| **NumPy**         | 轨迹数组操作                 | `data/`, `utils/`                     |
| **Pandas**        | 表格数据处理（可选）         | `data/`                               |
| **Matplotlib**    | 预测结果可视化               | `utils/visualize.py`                  |
| **PyYAML**        | 配置文件加载                 | `scripts/train.py`, `scripts/eval.py` |
| **Pillow**        | 图像场景图加载（可选）       | `data/dataset.py`                     |





如果你计划使用图像格式的场景语义图（`.png`/`.jpg`），则也需要安装 Pillow。对于默认的 `.npy` 场景图，则不需要该包。



来源: [README.md](https://zread.ai/README.md#L59-L67), [data/dataset.py](https://zread.ai/data/dataset.py#L34-L38)

## 流水线概述

整个工作流遵循严格的四个阶段顺序。每个阶段都会生成下一阶段所需的输入，因此跳过任何步骤都将导致下游出错。



每个阶段均由 `scripts/` 目录下的专用脚本驱动，且所有可调参数均位于 `configs/` 目录下的 YAML 配置文件中。首次运行时，使用 `configs/default.yaml` 的默认配置即可。

来源: [scripts/train.py](https://zread.ai/scripts/train.py#L70-L175), [scripts/eval.py](https://zread.ai/scripts/eval.py#L26-L90), [scripts/preprocess_eth_ucy.py](https://zread.ai/scripts/preprocess_eth_ucy.py#L141-L184)

## 第 1 步 — 数据预处理

ETH 和 UCY 数据集的原始格式异构：ETH 使用包含 8 列数据的 `obsmat.txt` 文件，而 UCY 使用需要进行线性插值的 `.vsp` 样条控制点文件。预处理脚本会将两者统一归一化为 `frame ped_id x y` 的文本格式，并将其拆分到 `train/` 和 `val/` 目录中。



```
python scripts/preprocess_eth_ucy.py \    --eth_dir data/ewap_dataset \    --ucy_dir data/crowds/data \    --out_dir data/processed
```

该脚本要求 `data/` 目录下保持仓库的原始目录结构——ETH 数据位于 `data/ewap_dataset/seq_eth/obsmat.txt` 和 `data/ewap_dataset/seq_hotel/obsmat.txt`，UCY 数据位于 `data/crowds/data/`。成功运行后，你将看到以下输出结构：



```
data/processed/
├── train/
│   ├── crowds_zara01.txt
│   ├── crowds_zara02.txt
│   ├── crowds_zara03.txt
│   ├── seq_eth.txt
│   ├── seq_hotel.txt
│   └── students001.txt
└── val/
    ├── students003.txt
    └── uni_examples.txt
```

默认的场景划分由 `--train_scenes` 和 `--val_scenes` 参数控制。内置默认值将 6 个场景分配给训练集，3 个场景分配给验证集，这与轨迹预测文献中常用的留一法评估协议相一致。

来源: [scripts/preprocess_eth_ucy.py](https://zread.ai/scripts/preprocess_eth_ucy.py#L141-L184), [scripts/preprocess_eth_ucy.py](https://zread.ai/scripts/preprocess_eth_ucy.py#L19-L33), [scripts/preprocess_eth_ucy.py](https://zread.ai/scripts/preprocess_eth_ucy.py#L62-L109)

## 第 2 步 — 训练模型

数据预处理完成后，使用单条命令即可启动训练。`--config` 参数指向一个 YAML 配置文件；`configs/default.yaml` 会训练**完整的 CSGAT-Net**（启用场景模块和社会模块）：



```
python scripts/train.py --config configs/default.yaml
```

在训练过程中，你会看到每个 epoch 的输出，如下所示：



```
epoch=0 loss=1.2345 val_ade=0.8721 val_fde=1.6543
epoch=1 loss=0.9876 val_ade=0.7432 val_fde=1.4321
...
```

训练循环会自动保存两个检查点：**`best.pt`**（验证集 ADE 最低）和 **`last.pt`**（最后一个 epoch）。两者均会被写入配置文件中 `train.save_dir` 所指定的目录。

### 核心配置参数

| 参数                   | 默认值 | 描述                           |
| ---------------------- | ------ | ------------------------------ |
| `data.obs_len`         | 8      | 观测轨迹长度（帧）             |
| `data.pred_len`        | 12     | 预测轨迹长度（帧）             |
| `data.neighbor_radius` | 2.0    | 邻居选择半径（米）             |
| `data.max_neighbors`   | 16     | 每个 Agent 的最大邻居数        |
| `model.hidden_dim`     | 64     | LSTM 与融合层的隐藏维度        |
| `model.scene_dim`      | 128    | 场景编码器输出维度             |
| `model.latent_dim`     | 16     | CVAE 潜在空间维度              |
| `model.use_scene`      | true   | 启用/禁用场景编码器            |
| `model.use_social`     | true   | 启用/禁用社会 GAT              |
| `train.batch_size`     | 32     | 训练批次大小                   |
| `train.epochs`         | 20     | 训练轮数                       |
| `train.lr`             | 0.001  | Adam 学习率                    |
| `train.beta`           | 1.0    | CVAE 损失中 KL 散度的权重      |
| `train.seed`           | 42     | 用于可复现性的随机种子         |
| `eval.sample_k`        | 20     | Best-of-K 采样中的 CVAE 样本数 |

两个布尔值参数——`model.use_scene` 和 `model.use_social`——是决定训练哪种模型变体的架构开关。这也是消融实验配置背后的核心机制。

来源: [scripts/train.py](https://zread.ai/scripts/train.py#L70-L175), [configs/default.yaml](https://zread.ai/configs/default.yaml#L1-L45), [models/csgat_net.py](https://zread.ai/models/csgat_net.py#L14-L55)

## 第 3 步 — 评估模型

训练完成后，在验证集上评估最佳检查点。评估脚本会加载模型，使用 **Best-of-K 采样**（默认 K=20）进行推理，并报告 ADE 和 FDE 指标：



```
python scripts/eval.py --config configs/default.yaml
```

预期输出格式：



```
ADE=0.5432 FDE=1.0278
```

指标计算方式如下：对于每个样本，模型从 CVAE 先验中生成 K 条轨迹预测，并选择其中最接近真实轨迹的那一条。这种 Best-of-K 协议是多模态轨迹预测评估中的标准做法。检查点路径从配置中的 `eval.checkpoint` 读取，默认为 `outputs/checkpoints/best.pt`。

来源: [scripts/eval.py](https://zread.ai/scripts/eval.py#L26-L90), [metrics/ade_fde.py](https://zread.ai/metrics/ade_fde.py)

## 第 4 步 — 可视化预测结果

生成将观测轨迹、真实轨迹和预测轨迹叠加显示的轨迹图。可视化脚本会将多条 CVAE 采样路径渲染为半透明的绿色线条，从而直观地展示模型的多模态输出：



```
python scripts/visualize.py --config configs/default.yaml --index 0
```

`--index` 参数用于选择要可视化的验证样本（从 0 开始索引）。输出将保存至 `outputs/figures/sample_<index>.png`。该脚本会自动逆转变换数据集的归一化操作（中心化 + 缩放），从而使绘制的坐标位于原始的世界空间中。

![样本预测可视化](https://github.com/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/blob/main/outputs/figures/sample_0.png?raw=true)

来源: [scripts/visualize.py](https://zread.ai/scripts/visualize.py#L25-L93), [utils/visualize.py](https://zread.ai/utils/visualize.py#L9-L42)

## 消融实验

本仓库附带了四个专为消融实验准备的现成配置文件，分别切换了 `use_scene` 和 `use_social` 参数：

| 配置文件                         | `use_scene` | `use_social` | 检查点目录                      |
| -------------------------------- | :---------: | :----------: | ------------------------------- |
| `configs/ablation_baseline.yaml` |      ✗      |      ✗       | `outputs/checkpoints/baseline/` |
| `configs/ablation_scene.yaml`    |      ✓      |      ✗       | `outputs/checkpoints/scene/`    |
| `configs/ablation_social.yaml`   |      ✗      |      ✓       | `outputs/checkpoints/social/`   |
| `configs/ablation_full.yaml`     |      ✓      |      ✓       | `outputs/checkpoints/full/`     |

将 `--config` 指向相应的文件即可运行每种消融变体：



```
# 基线：仅轨迹 LSTMpython scripts/train.py --config configs/ablation_baseline.yaml # 仅场景：LSTM + 场景编码器python scripts/train.py --config configs/ablation_scene.yaml # 仅社会：LSTM + 图注意力python scripts/train.py --config configs/ablation_social.yaml # 完整模型：LSTM + 场景 + 社会python scripts/train.py --config configs/ablation_full.yaml
```

每个配置都会将结果写入各自的检查点目录，因此你可以依次运行全部四个实验而不会覆盖之前的结果。

来源: [configs/ablation_baseline.yaml](https://zread.ai/configs/ablation_baseline.yaml#L17-L24), [configs/ablation_scene.yaml](https://zread.ai/configs/ablation_scene.yaml#L17-L24), [configs/ablation_social.yaml](https://zread.ai/configs/ablation_social.yaml#L17-L24), [configs/ablation_full.yaml](https://zread.ai/configs/ablation_full.yaml#L17-L24)

## 故障排除

| 症状                              | 可能原因                                 | 解决方案                                                     |
| --------------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| 预处理时报 `FileNotFoundError`    | ETH/UCY 原始数据不在预期路径中           | 确认 `data/ewap_dataset/` 和 `data/crowds/data/` 存在且目录结构正确 |
| `ImportError: PyYAML is required` | 缺少 PyYAML 包                           | `pip install pyyaml`                                         |
| 训练在 CPU 而非 GPU 上运行        | `torch.cuda.is_available()` 返回 `False` | 检查 CUDA 安装；脚本会自动回退到 CPU                         |
| 预处理后 `data/processed/` 为空   | 在源目录中未找到匹配的文件               | 确认输入目录下存在 `obsmat.txt` 和 `.vsp` 文件               |
| 可视化未显示场景背景              | 配置中的 `data.scene_dir` 为 `null`      | 提供场景语义图（`.npy`）并将 `scene_dir` 设置为其所在文件夹  |

来源: [scripts/train.py](https://zread.ai/scripts/train.py#L78-L79), [data/dataset.py](https://zread.ai/data/dataset.py#L22-L38), [scripts/preprocess_eth_ucy.py](https://zread.ai/scripts/preprocess_eth_ucy.py#L143-L145)

## 后续步骤

你现在已拥有一个可运行的端到端流水线。若要理解*为什么*每个组件会如此运作，以及如何为你自己的研究自定义它们，请深入探索以下详细文档：

1. **理解架构** → 从[架构概述](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/3-architecture-overview)开始建立全局观念，然后深入各个独立模块：[轨迹 LSTM 编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/4-trajectory-lstm-encoder)、[社会图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network)、[ResNet 场景语义编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/6-resnet-scene-semantic-encoder) 和 [CVAE 多模态轨迹预测头](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/7-cvae-multimodal-trajectory-head)。
2. **自定义数据流水线** → 阅读 [ETH/UCY 数据集预处理](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/9-eth-ucy-dataset-preprocessing) 和 [轨迹数据集与邻居构建](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/10-trajectory-dataset-and-neighbor-construction)，详细了解原始文件是如何转化为训练张量的。
3. **调整训练与评估** → 阅读 [CVAE 损失函数](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/13-cvae-loss-function) 和 [基于 Best-of-K 的 ADE/FDE 指标](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/14-ade-fde-metrics-with-best-of-k)，理解这些数值的实际含义。
4. **运行消融实验** → [配置与消融设置](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/15-configuration-and-ablation-setup) 深入解释了基于配置文件的模块切换机制。



# 架构概述

本页展示了 **CSGATNet** 的端到端架构——这是一个行人轨迹预测模型，融合了三种互补的信号来源：**个体运动历史**、**相邻行人间的社交交互**，以及来自分割图的**场景语义上下文**。该模型通过条件变分自编码器（CVAE）生成多模态的未来轨迹，从而能够捕捉人类运动的固有不确定性。理解该架构是阅读本文档后续各组件深入解析的先决条件。

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L14-L54), [README.md](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/README.md#L40-L45)

------

## 设计理念

CSGATNet 基于**模块化三分支融合**原则构建：每个信息通道（轨迹、社交、场景）独立编码，随后合并为一个统一的条件向量，以驱动 CVAE 和自回归解码器。此设计带来了两个关键特性。首先，任何分支均可通过单个布尔标志（`use_scene`、`use_social`）禁用，这正是无需代码重复即可生成四种消融实验配置的方法。其次，融合维度会自动适配——在构造时，`fuse_dim` 的计算会将 `hidden_dim`（始终存在，来自轨迹编码器）加上 `scene_dim` 和/或 `hidden_dim`，具体取决于哪些分支处于激活状态 [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L40-L44)。下表总结了由此产生的四种运行模式：

| 配置文件                              | `use_scene` | `use_social` | 融合维度 | 激活分支    |
| ------------------------------------- | ----------- | ------------ | -------- | ----------- |
| `ablation_baseline.yaml`              | `false`     | `false`      | 64       | 仅轨迹      |
| `ablation_social.yaml`                | `false`     | `true`       | 128      | 轨迹 + 社交 |
| `ablation_scene.yaml`                 | `true`      | `false`      | 192      | 轨迹 + 场景 |
| `ablation_full.yaml` / `default.yaml` | `true`      | `true`       | 256      | 三者全部    |

来源: [ablation_baseline.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_baseline.yaml#L17-L23), [ablation_social.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_social.yaml#L17-L23), [ablation_scene.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_scene.yaml#L17-L23), [ablation_full.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_full.yaml#L17-L23), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L17-L23)

------

## 端到端数据流

下图追踪了单个样本从原始轨迹输入，经过完整前向传播，直至输出预测未来路径的全过程。请注意**三个编码器分支**在拼接点之前是如何并行运作的，以及在 CVAE 头部发生的**训练/推断分流**——训练期间进行后验采样，而推断期间则进行带有 K 次重复的先验采样。



来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L85-L126)

------

## 核心组件概要

上图中的每个模块都是一个职责明确、相互独立的 PyTorch 模块。本表为五个核心组件提供了简明的参考信息——包括它们的类名、源文件、输入/输出形状（针对默认配置：`hidden_dim=64`、`scene_dim=128`、`latent_dim=16`），以及在流水线中的功能角色。

| 组件           | 类                     | 来源                                                         | 输入形状                          | 输出形状              | 角色                                |
| -------------- | ---------------------- | ------------------------------------------------------------ | --------------------------------- | --------------------- | ----------------------------------- |
| 轨迹编码器     | `LSTMEncoder`          | [lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py) | `(B, 8, 2)`                       | `(B, 64)`             | 将观测运动编码为隐状态向量          |
| 邻居编码器     | `LSTMEncoder` (共享类) | [lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py) | `(B×16, 8, 2)`                    | `(B, 16, 64)`         | 独立编码每个邻居的轨迹              |
| 社交图注意力   | `SocialGAT`            | [gat.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/gat.py) | agent `(B,64)`, neigh `(B,16,64)` | `(B, 64)`             | 通过掩码感知的 softmax 关注邻居特征 |
| 场景语义编码器 | `SceneEncoder`         | [resnet_seg.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/resnet_seg.py) | `(B, 3, H, W)`                    | `(B, 128)`            | ResNet 主干 → 全局池化 → 线性投影   |
| CVAE 头部      | `CVAEHead`             | [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py) | cond `(B,64)`, fut `(B,64)`       | `z (B,16)` + 分布参数 | 带有重参数化技巧的后验/先验网络     |

来源: [lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py#L7-L23), [gat.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/gat.py#L7-L30), [resnet_seg.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/resnet_seg.py#L9-L44), [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L7-L42)

------

## CSGATNet 编排器

`CSGATNet` 是顶层的 `nn.Module`，负责实例化、连接并编排所有子模块。其 `forward` 方法在三个逻辑阶段内实现了完整的流水线：

**阶段 1 — 并行编码。** 观测轨迹通过 `traj_encoder` 生成 `agent_feat`。同时，邻居轨迹被展平并通过 `_encode_neighbors` 进行批量编码，随后由 `social_gat` 进行聚合。场景图在可用时，会通过 `scene_encoder` 处理。如果任何分支被禁用，则会用零张量或仅用智能体特征进行替换 [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L95-L108)。

**阶段 2 — CVAE 采样。** 拼接并融合后的 `cond_feat` 进入 CVAE。在训练期间（`fut_traj is not None`），后验网络同时以 `cond_feat` 和未来轨迹编码为条件，从而产生更紧凑的潜变量分布。在推断期间，仅使用先验网络，并重复采样 `sample_k` 次以生成多模态预测 [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L112-L126)。

**阶段 3 — 自回归解码。** 初始解码器状态 `(h₀, c₀)` 由 `[cond_feat, z]` 通过 `dec_init` 导出，该方法将一个线性投影向量拆分为两半。在 12 个预测步的每一步中，前一个位置被嵌入并输入 LSTM，输出再通过 `out_fc` 产生位移 `Δ`。新位置为 `prev + Δ`，它将作为下一步的输入反馈回去 [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L67-L83)。





自回归解码器会累积位移误差——因此，由 `[cond_feat, z]` 初始化的 `(h₀, c₀)` 的质量，是影响长期 FDE 最敏感的因素。在调试预测漂移时，请在分块拆分前检查 `dec_init` 输出的范数。



来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L85-L126), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L67-L83)

------

## 训练与推断分流

`CSGATNet` 的 `forward` 方法包含一个显式分支，根据是否提供了 `fut_traj` 来切换行为。这是标准的 CVAE 教师强制模式，准确理解它非常重要，因为损失计算正依赖于此。

| 方面            | 训练（提供 `fut_traj`）                            | 推断（`fut_traj=None`）                   |
| --------------- | -------------------------------------------------- | ----------------------------------------- |
| CVAE 分支       | 后验 `q(z                                          | x,y)`                                     |
| 潜变量 `z` 来源 | 从后验 `(μ_q, σ²_q)` 中采样                        | 从先验 `(μ_p, σ²_p)` 中采样               |
| 预测数量        | 1                                                  | `sample_k`（默认 20）                     |
| 输出形状        | `(B, 12, 2)`                                       | `(B, K, 12, 2)`                           |
| 返回值          | `pred, prior_μ, prior_logvar, post_μ, post_logvar` | `pred, prior_μ, prior_logvar, None, None` |
| 损失            | MSE 重构 + β·KL(后验 ∥ 先验)                       | 不适用（无梯度）                          |

在训练期间，KL 散度项将后验向先验进行正则化，使得在推断时，仅从先验采样就能产生有意义的潜变量编码。`beta` 参数（在 [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L30) 中默认为 1.0）控制着这种权衡——可以应用 β-VAE 调度来实现解耦。

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L112-L126), [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L34-L42), [cvae_loss.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/losses/cvae_loss.py#L22-L33), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L28-L31)

------

## 项目模块映射

仓库的组织方式使得每个 Python 模块对应一个架构关注点。下方视图展示了源目录与上述流水线阶段之间的关系。



```
Flag64Zhang/Pedestrian-Trajectory-Prediction---Graph-Attention/
│
├── models/                          │   ├── csgat_net.py                 #    CSGATNet 编排器（融合 + 解码）
│   ├── lstm_encoder.py              #    轨迹与邻居 LSTM 编码器
│   ├── gat.py                       #    SocialGAT 注意力层
│   ├── resnet_seg.py                #    基于 ResNet 的场景编码器
│   └── cvae_head.py                 #    CVAE 后验/先验 + 重参数化
│
├── data/                            # 📊 数据加载与预处理
│   ├── dataset.py                   #    TrajectoryDataset, Sample, collate_fn
│   ├── transforms.py                #    TrajectoryRotateScale 数据增强
│   └── ewap_dataset/                #    原始 ETH/UCY 源文件
│
├── losses/                          # 📉 损失函数
│   └── cvae_loss.py                 #    MSE 重构 + KL 散度
│
├── metrics/                         # 📏 评估指标
│   └── ade_fde.py                   #    带有 Best-of-K 选择的 ADE/FDE
│
├── scripts/                         # 🚀 入口脚本
│   ├── train.py                     #    训练循环 + 检查点保存
│   ├── eval.py                      #    独立评估脚本
│   └── visualize.py                 #    预测可视化
│
├── configs/                         # ⚙️ YAML 配置
│   ├── default.yaml                 #    完整模型（场景 + 社交）
│   ├── ablation_baseline.yaml       #    仅轨迹
│   ├── ablation_social.yaml         #    轨迹 + 社交
│   ├── ablation_scene.yaml          #    轨迹 + 场景
│   └── ablation_full.yaml           #    完整模型（与 default 相同）
│
├── utils/                           # 🔧 实用工具
│   ├── io.py                        #    检查点保存/加载，目录辅助
│   ├── seed.py                      #    可复现性随机种子设置
│   └── visualize.py                 #    Matplotlib 轨迹绘制
│
└── outputs/                         # 📦 产出物（git 忽略）
    ├── checkpoints/                 #    训练好的模型权重
    └── figures/                     #    预测可视化 PNG 图像
```

来源: 跨所有目录的仓库结构分析

------

## 配置到架构的绑定

`CSGATNet` 的每个构造函数参数均由 YAML 配置文件驱动，`train.py` 脚本充当两者之间的接线层。这种映射是直接且透明的——模型中没有任何与配置不一致的隐藏默认值 [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L119-L127)。下表展示了完整的参数绑定：

| YAML 路径          | 模型构造函数参数 | 默认值 | 描述                        |
| ------------------ | ---------------- | ------ | --------------------------- |
| `data.obs_len`     | `obs_len`        | 8      | 观测时间步数                |
| `data.pred_len`    | `pred_len`       | 12     | 需预测的未来时间步数        |
| `model.hidden_dim` | `hidden_dim`     | 64     | LSTM 隐层大小及融合输出维度 |
| `model.scene_dim`  | `scene_dim`      | 128    | 场景编码器输出维度          |
| `model.latent_dim` | `latent_dim`     | 16     | CVAE 潜变量空间维度         |
| `model.use_scene`  | `use_scene`      | `true` | 启用/禁用 ResNet 场景分支   |
| `model.use_social` | `use_social`     | `true` | 启用/禁用 SocialGAT 分支    |





当 `use_scene=false` 且 `use_social=false` 时，模型退化为一个纯粹的 LSTM→CVAE 基线——这正是 `ablation_baseline.yaml` 所配置的内容。此时 `fuse` 层仅接收 `agent_feat`（维度=64），并将其投影通过 `Linear(64, 64) → ReLU`，这实际上是一个恒等缩放变换加上非线性。



来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L15-L24), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L119-L127), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L17-L23)

------

## 接下来去哪

架构概述已建立起宏观结构。上表列出的每个组件都有其专属页面，包含实现细节、数学基础和设计依据。推荐的阅读顺序遵循从输入到输出的数据流：

1. **[轨迹 LSTM 编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/4-trajectory-lstm-encoder)** —— 观测运动历史如何被压缩为固定大小的向量，以及为何选择最终隐状态而非序列池化。
2. **[社交图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network)** —— 掩码感知注意力机制，如何在不产生填充伪影的情况下聚合可变数量的邻居特征。
3. **[ResNet 场景语义编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/6-resnet-scene-semantic-encoder)** —— ResNet 主干特征提取、全局池化策略，以及使用冻结与微调主干的设计决策。
4. **[CVAE 多模态轨迹头部](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/7-cvae-multimodal-trajectory-head)** —— 后验/先验网络分离、重参数化技巧，以及多模态如何从潜变量采样中产生。
5. **[特征融合与自回归解码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/8-feature-fusion-and-autoregressive-decoder)** —— 基于拼接的融合策略和累积位移的解码器循环。

对于更关注数据和训练流水线的读者，可直接跳转至 [ETH/UCY 数据集预处理](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/9-eth-ucy-dataset-preprocessing) 或 [训练循环与检查点](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/12-training-loop-and-checkpointing)。



# 轨迹 LSTM 编码器

**轨迹 LSTM 编码器** 是 CSGAT-Net 架构中的基础特征提取模块——负责将原始的 2D 坐标序列转换为紧凑的隐状态表示，供下游模块（社交注意力、场景融合、CVAE 采样）进行推理。尽管其代码极为精简（仅 24 行），该编码器却体现了一项深思熟虑的设计：在循环处理*之前*，通过一个线性投影层将低维空间坐标提升至模型的表示空间，确保 LSTM 在特征而非原始位置上进行运算。同一个 `LSTMEncoder` 类在完整模型中被实例化了两次——一次用于目标行人，一次用于每个邻居——这使其成为整个流水线中复用率最高的构建模块。

来源：[lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py#L1-L24), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L34-L35)

## 架构与数据流

该编码器遵循两阶段流水线：**线性投影 → LSTM 循环 → 隐状态提取**。每个时间步的原始轨迹坐标 `(x, y)` 首先通过 `input_fc` 投影到隐维度空间，然后由单层 LSTM 进行处理。仅返回最后一层 LSTM 的最终隐状态作为轨迹的特征向量——所有中间输出均被丢弃。此设计将 LSTM 纯粹视为序列摘要器，将可变长度的观测窗口压缩为单个固定大小的嵌入。



核心洞见在于，`input_fc` 执行了从 2 到 64（默认 `hidden_dim`）的**维度扩展**，在 LSTM 处理之前将每个坐标对转换为丰富的特征向量。若无此投影，LSTM 将需要同时学习空间到特征的映射*以及*时间动态——这将导致更困难的优化空间。该投影清晰地分离了这两项关注点。

来源：[lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py#L10-L23)

## 模块规格

`LSTMEncoder` 类暴露了四个可配置参数，均具有与项目标准配置相匹配的合理默认值：

| 参数         | 类型    | 默认值 | 描述                                                         |
| ------------ | ------- | ------ | ------------------------------------------------------------ |
| `input_dim`  | `int`   | `2`    | 每个输入时间步的维度（x, y 坐标）。与数据集中的 `obs_traj[:, :, 0:2]` 切片相匹配。 |
| `hidden_dim` | `int`   | `64`   | 投影目标维度及 LSTM 隐层大小。所有下游模块（GAT、融合层、CVAE）均依赖此值。 |
| `num_layers` | `int`   | `1`    | 堆叠的 LSTM 层数。单层足以应对 8 时间步的观测窗口。          |
| `dropout`    | `float` | `0.0`  | LSTM 层间的 Dropout。仅在 `num_layers > 1` 时生效；内部由条件语句守护。 |

内部层组合为：

| 层         | 形状                                           | 用途                           |
| ---------- | ---------------------------------------------- | ------------------------------ |
| `input_fc` | `Linear(2, 64)`                                | 将原始坐标提升至特征空间       |
| `lstm`     | `LSTM(64, 64, num_layers=1, batch_first=True)` | 时序序列编码                   |
| 输出       | `h[-1]` → 形状 `(batch, 64)`                   | 最后一层的隐状态，作为轨迹嵌入 |





`nn.LSTM` 中的 `dropout` 参数应用在堆叠的层**之间**，而非单层内部。由于默认 `num_layers=1`，dropout 会通过条件判断 `dropout if num_layers > 1 else 0.0` 自动置零——否则在单层情况下尝试设置 `dropout > 0` 将引发 PyTorch 运行时错误。



来源：[lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py#L7-L17)

## 前向传播机制

前向方法接收一个轨迹张量并返回单个隐状态向量。对于任何集成或调试此模块的开发者而言，理解张量形状约定和隐状态提取逻辑至关重要：

**输入**：`traj` — 形状 `(batch_size, obs_len, 2)` — 一批观测轨迹，每条由 8 个时间步的归一化 `(x, y)` 坐标组成（以最后观测点为中心并按最大绝对值进行缩放，如 [TrajectoryDataset](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L163-L176) 所执行的操作）。

**处理**：通过 PyTorch 的广播机制，`input_fc` 投影在**每个时间步独立应用**——`nn.Linear` 作用于最后一个维度，因此 `(B, T, 2)` 被投影为 `(B, T, hidden_dim)` 而无需任何手动循环。随后，LSTM 按时间顺序处理此序列，在所有 8 个时间步中累积隐状态。

**输出**：`h[-1]` — 形状 `(batch_size, hidden_dim)` — **最后时间步**的**最后一层 LSTM** 的隐状态。来自 `nn.LSTM` 的 `h` 张量形状为 `(num_layers, batch_size, hidden_dim)`，`h[-1]` 选取了最顶层的的状态。当 `num_layers=1` 时，这即为单层的最终隐状态。



```
def forward(self, traj: torch.Tensor) -> torch.Tensor:    x = self.input_fc(traj)            _, (h, _) = self.lstm(x)       # h: (num_layers, B, hidden_dim)    return h[-1]                    # (B, hidden_dim)
```

单元状态 `c` 被显式丢弃（`_`）。这是有意为之——编码器的职责是生成**摘要向量**，而非将循环状态传递给解码器。CSGATNet 中的解码器从融合的条件向量和潜变量采样中初始化其自身的隐状态，而非来自编码器的单元记忆。

来源：[lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py#L20-L23), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L67-L71)

## CSGATNet 中的双重实例化

`LSTMEncoder` 在 [CSGATNet](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L14) 中并非使用一次，而是**两次**，担当结构相同但语义不同的角色：



```
self.traj_encoder = LSTMEncoder(input_dim=2, hidden_dim=hidden_dim)   # 目标行人self.neigh_encoder = LSTMEncoder(input_dim=2, hidden_dim=hidden_dim)  # 邻居行人
```

| 实例            | 变量          | 用途                    | 输出用法                                             |
| --------------- | ------------- | ----------------------- | ---------------------------------------------------- |
| `traj_encoder`  | `agent_feat`  | 编码目标行人的 8 步观测 | 直接拼接到融合向量中；亦作为社交注意力中的查询       |
| `neigh_encoder` | `neigh_feats` | 编码每个邻居的 8 步观测 | 输入至 `SocialGAT` 作为键/值特征，用于注意力加权聚合 |

对于邻居编码，`_encode_neighbors` 方法将批处理的邻居张量从 `(batch, num_neigh, obs_len, 2)` 重塑为扁平的 `(batch × num_neigh, obs_len, 2)` 格式，在单次前向传播中将所有邻居经由 `neigh_encoder` 处理，然后重塑回 `(batch, num_neigh, hidden_dim)`。这是一种效率优化——编码器将它们作为扩展的批维度进行处理，而非遍历邻居循环。



两个实例共享相同的架构（`input_dim=2, hidden_dim=64`），但维持**独立的权重**。这很重要——目标编码器学习以优化融合及 CVAE 条件路径的方式来表示轨迹，而邻居编码器学习专为社交注意力评分调整的表示。此处未采用权重共享，从而赋予每个编码器自由特化的空间。





尽管两个编码器具有相同的超参数，它们**并不**共享权重。如果你检查 `model.traj_encoder.lstm.weight_ih_l0` 和 `model.neigh_encoder.lstm.weight_ih_l0`，它们在训练后会发散。这种分离使得目标编码器能够专注于对下游预测有用的轨迹特征，而邻居编码器可以强调有助于区分交互模式的特征——相同架构下承受不同的优化压力。



来源：[csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L34-L35), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L57-L64), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L95-L107)

## 输入预处理上下文

到达 LSTM 编码器的轨迹坐标**并非**原始的世界空间位置。`TrajectoryDataset` 应用了两步归一化操作，直接影响编码器的学习内容：

1. **中心化**：所有坐标均被平移，使最后一个观测点成为原点（`obs = obs - obs[-1]`）。这意味着编码器看到的轨迹总是近似结束于 `(0, 0)`，消除了绝对位置偏差，并迫使模型学习**运动模式**而非记忆位置。
2. **缩放**：中心化后，所有坐标除以跨观测、未来及邻居轨迹的最大绝对值（`obs = obs / max_val`）。这会将数值映射至约 `[-1, 1]` 的范围内，使 LSTM 输入保持在良态区间内，以确保梯度稳定流动。

这些预处理步骤对编码器的有效性至关重要。若无中心化，LSTM 将需要学习绝对位置无关性——这是一种浪费且更困难的优化。若无缩放，在不同场景（如 ETH 与 hotel 数据集）中，坐标幅度可能相差数个数量级，从而导致不稳定的隐状态动态。

来源：[dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L163-L176)

## 消融 significance

LSTM 编码器是项目中**唯一在所有消融配置中均存留的组件**。即使是禁用了场景和社交模块的基线配置（`ablation_baseline.yaml`），也保留了轨迹编码器作为唯一的特征提取器。这使其成为衡量所有其他模块贡献的架构常数：

| 配置     | `use_scene` | `use_social` | 编码器角色                                                   |
| -------- | ----------- | ------------ | ------------------------------------------------------------ |
| 基线     | `false`     | `false`      | **唯一**特征源；仅 `agent_feat` 被融合并输入至 CVAE          |
| 仅场景   | `true`      | `false`      | 与 `SceneEncoder` 配对；融合维度 = `hidden_dim + scene_dim`  |
| 仅社交   | `false`     | `true`       | 与 `SocialGAT` 配对；融合维度 = `hidden_dim + hidden_dim`    |
| 完整模型 | `true`      | `true`       | 三个流全部启用；融合维度 = `hidden_dim + scene_dim + hidden_dim` |

在基线消融中，由于没有附加流，`fuse_dim = hidden_dim = 64`，融合层退化为 `Linear(64, 64)`。编码器的输出几乎直接传递至 CVAE 条件路径，使其表示质量成为影响基线性能的唯一最关键因素。

来源：[ablation_baseline.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_baseline.yaml#L17-L23), [ablation_full.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_full.yaml#L17-L23), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L40-L46)

## 设计原理与权衡

编码器的设计反映了若干深思熟虑的权衡，任何考虑对其进行修改的开发者都值得了解：

**为什么使用 `input_fc` 而非让 LSTM 直接处理原始 2D 输入？** 采用 `input_size=2` 和 `hidden_size=64` 的 LSTM 理论上可行，但其输入到隐层变换的有效权重矩阵仅为 `4 × 64 × 2 = 512` 个参数——不足以从仅有的两个坐标中学习丰富的空间特征。通过首先投影至 64 维，LSTM 的输入到隐层矩阵变为 `4 × 64 × 64 = 16,384` 个参数，为时间模式识别提供了大得多的容量。

**为什么仅使用 `h[-1]` 而非完整输出序列？** 观测窗口较短（8 个时间步），且预测任务是全局性的——我们需要的是关于*行人曾去过何处及正前往何方*的摘要，而非逐步的时间步特征图。返回最终隐状态是序列到向量编码的标准方法，且符合下游融合层对每个行人期望单个向量的设定。基于注意力的解码器可能需要序列输出，但 CSGATNet 使用的是自回归 LSTM 解码器。

**为什么使用单层 LSTM？** 观测长度仅为 8 个时间步——单层 LSTM 已具备足够的感受野来捕获相关的时间依赖性。增加层数会增加参数和训练复杂度，而对如此短的序列收益递减。所有消融配置默认使用 `num_layers=1`，且默认配置保持 `hidden_dim=64`，表明该模型被设计为轻量级。

来源：[lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py#L7-L23), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L17-L23)

## 与下游模块的集成

编码器的输出 `agent_feat`（形状 `(B, 64)`）在 CSGATNet 中流入三个下游路径，每个路径对其消耗方式不同：



1. **特征融合**（`self.fuse`）：`agent_feat` 与场景和/或社交特征拼接，然后投影回 `hidden_dim`。这是主要路径——编码器的输出成为条件向量 `cond_feat` 的主干，驱动 CVAE 先验/后验以及解码器初始化。
2. **SocialGAT 查询**：当启用社交注意力时，`agent_feat` 被扩展并与每个邻居的特征向量拼接以计算注意力分数。它充当决定哪些邻居最相关的“查询”。完整注意力机制详见[社交图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network)。
3. **解码器初始化**（间接）：通过融合层和 CVAE，编码器的表示最终决定了自回归解码器的初始隐状态 `(h₀, c₀)`。这意味着编码器轨迹摘要的质量直接决定了解码器生成准确未来路径的能力。

来源：[csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L95-L116), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L67-L71)

## 后续内容

轨迹 LSTM 编码器生成了核心的行人表示，但其输出仅是完整 CSGAT-Net 流水线中的一个流。深入解析的后续页面将涵盖剩余的特征提取与融合模块：

- **[社交图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network)** — 来自邻居编码器的 `neigh_feats` 如何利用 Agent 特征作为查询，进行注意力加权聚合成 `social_feat`。
- **[ResNet 场景语义编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/6-resnet-scene-semantic-encoder)** — 场景图像如何被编码为 `scene_feat` 并与轨迹嵌入进行融合。
- **[特征融合与自回归解码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/8-feature-fusion-and-autoregressive-decoder)** — `agent_feat`、`social_feat` 和 `scene_feat` 如何被拼接、投影，并用于初始化预测解码器。

# 社交图注意力网络

**社交图注意力网络**（SocialGAT）是 CSGAT-Net 架构中负责建模行人与行人之间交互的模块。它通过掩码加性注意力机制，将可变长度的邻居轨迹集合转换为单一固定维度的**社交特征向量**——使得模型能够区分哪些附近行人最影响目标 Agent 的未来运动，而无需依赖人工设计的交互规则或稠密邻接矩阵。

## 从邻居到注意力：核心思想

在真实的拥挤场景中，行人的轨迹并非受周围所有人同等程度的影响，而是由特定的一部分人决定——直接走入其路线的人，远比几米外平行行走的人重要得多。SocialGAT 通过**可学习的注意力机制**将这一直觉转化为实际操作，为编码轨迹特征与目标 Agent 最相关的邻居分配更高的权重。该模块在已经编码的轨迹表示（由共享的 [Trajectory LSTM 编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/4-trajectory-lstm-encoder) 生成）上进行操作，因此它推理的是**学习到的运动特征**，而非原始坐标。

下图展示了 SocialGAT 如何融入 CSGAT-Net 更广泛的特征提取流程中：



来源: [gat.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/gat.py#L7-L30), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L34-L37), [lstm_encoder.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/lstm_encoder.py#L7-L23)

## 架构深入剖析

### SocialGAT 模块

`SocialGAT` 类实现了一个单头加性注意力机制——也被称为 **Bahdanau-style 注意力**——该机制专为可变数量的邻居集合而适配。其设计做出了三个刻意的选择：加性（而非点积）评分、显式的掩码感知归一化，以及带有 `tanh` 非线性激活的双层评分 MLP。

| 组件             | 形状变换                   | 用途                                  |
| ---------------- | -------------------------- | ------------------------------------- |
| `fc` (Linear)    | `(B, K, 128) → (B, K, 32)` | 将拼接的 Agent-邻居对投影到注意力空间 |
| `score` (Linear) | `(B, K, 32) → (B, K, 1)`   | 为每个邻居生成标量相关性得分          |
| Masked Softmax   | `(B, K) → (B, K)`          | 仅在有效（非填充）邻居上进行归一化    |
| Weighted Sum     | `(B, K, 64) → (B, 64)`     | 按注意力权重聚合邻居特征              |

前向传播分为四个阶段进行：

**1. 配对构建。** 形状为 `(B, 64)` 的 Agent 特征 `agent_feat` 被扩展并与每个邻居特征拼接，生成形状为 `(B, K, 128)` 的配对表示。这确保了注意力得分是**关系型**的——它同时取决于 Agent 的状态和每个邻居的状态，而不是仅取决于其中一方。

**2. 注意力评分。** 配对特征依次经过 `tanh(fc(x))` 和 `score(...)`，为每个邻居生成一个原始标量。`tanh` 将中间激活限制在 `[-1, 1]` 之间，防止评分函数中的梯度爆炸，同时保留方向信息。

**3. 掩码感知归一化。** 这是最微妙的步骤。填充的邻居（超出给定 Agent 实际邻居数量的部分）在 `neigh_mask` 中用 `0` 标记。代码依次执行三个操作：`masked_fill` 在 softmax 之前将掩码位置替换为 `-1e9`，确保它们获得接近零的概率；接着 `weights * mask` 提供了额外的安全层，将任何残余清零；最后，重新归一化 `weights / (weights.sum(...) + 1e-8)` 保证权重仅在有效邻居上求和为 1。这种三重保护方法无论存在多少邻居，都能确保**严格的数值正确性**。

**4. 加权聚合。** 归一化后的权重与每个邻居的特征向量相乘，并在邻居维度上求和，生成单一的 `B × 64` 社交特征。





`attn_dim=32` 的默认值特意设置得比 `hidden_dim=64` 小，形成了一个信息瓶颈，迫使注意力机制学习一个紧凑的“相关性子空间”。如果你显著增加了 `max_neighbors`，请考虑按比例缩放 `attn_dim` 以保持表征能力。



来源: [gat.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/gat.py#L7-L30)

### 邻居编码流程

在 SocialGAT 能够对邻居进行注意力操作之前，必须对它们的原始轨迹进行编码。`CSGATNet` 中的 `_encode_neighbors` 方法通过**将批次和邻居维度展平为单一维度**，运行共享的 `LSTMEncoder`，然后再重塑回来，以此处理该过程：



```
Input:  neigh_traj  (B, K, 8, 2)
        ↓ view(B*K, 8, 2)
Flat:   (B*K, 8, 2)
        ↓ LSTMEncoder
Feats:  (B*K, 64)
        ↓ view(B, K, 64)
Output: (B, K, 64)
```

这种设计在 Agent 轨迹和所有邻居轨迹之间（通过 `self.neigh_encoder`）共享单个 `LSTMEncoder` 实例，这有两个重要意义：模型学习到一种**统一的轨迹表示**，其中 Agent 和邻居特征位于相同的嵌入空间中，使得后续的注意力比较在语义上具有意义；并且提高了参数效率——不需要维护独立的编码器，相同的权重处理所有轨迹。





尽管 `neigh_encoder` 与 `traj_encoder` 共享相同的 `LSTMEncoder` 类，但它们是**分别实例化**的（`self.traj_encoder` 和 `self.neigh_encoder`），因此它们的权重并不绑定。如果你想要真正的权重共享以强制实现相同的嵌入空间，请在 `_encode_neighbors` 中将 `self.neigh_encoder` 替换为 `self.traj_encoder`。



来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L57-L64), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L34-L35)

## 邻居的构建方式

社交注意力的质量在很大程度上取决于哪些行人被作为邻居输入。数据流水线在样本构建时使用两个参数来确定这一点：

| 参数              | 默认值   | 作用                                   |
| ----------------- | -------- | -------------------------------------- |
| `neighbor_radius` | 2.0 (米) | 最后观测帧的欧几里得距离阈值           |
| `max_neighbors`   | 16       | 邻居数量的硬性上限（超出部分会被截断） |

如果 `‖pos_n(t_obs_last) − pos_p(t_obs_last)‖ ≤ neighbor_radius`，则行人 `n` 符合成为目标 Agent `p` 的邻居的条件。这种**空间邻近性启发式方法**在最后一个观测帧计算一次，然后检索邻居的**完整 8 帧观测轨迹**进行编码。收集完毕后，邻居按距离排序（隐含在迭代顺序中）并截断至 `max_neighbors`。在数据集中，缺失的邻居用零填充，并用 `neigh_mask[i] = 0.0` 标记，SocialGAT 的掩码 softmax 会优雅地忽略这些标记。

来源: [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L84-L93), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L11-L12)

## 集成至 CSGAT-Net

在完整模型中，社交特征是融合到 CVAE 条件向量中的三个特征流之一。`CSGATNet` 的前向传播按如下方式组装特征：



```
agent_feat = self.traj_encoder(obs_traj)          feats = [agent_feat] if self.use_social:    neigh_feats = self._encode_neighbors(...)      # (B, K, 64)    social_feat = self.social_gat(agent_feat, ...) # (B, 64)    feats.append(social_feat) cond_feat = self.fuse(torch.cat(feats, dim=-1))    # (B, 64)
```

当 `use_social=True` 且 `use_scene=True` 时，融合输入维度为 `64 + 128 + 64 = 256`，通过带有 ReLU 的线性层投影回 `hidden_dim=64`。当 `use_social=True` 且 `use_scene=False` 时，该维度变为 `64 + 64 = 128`。通过配置标志实现的**条件门控**使得社交模块可以完全移除——这是一个促成消融实验的关键设计选择。

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L85-L110)

## 消融实验证据：为何社交注意力至关重要

该项目的消融实验分离了每个模块的贡献。结果证实，社交交互建模在 ADE 上提供了最显著的单模态改进，并且将其与场景特征结合产生了最佳的 FDE：

| 模型变体           | `use_social` | `use_scene` | ADE ↓      | FDE ↓      |
| ------------------ | ------------ | ----------- | ---------- | ---------- |
| 基线 (仅 LSTM)     | ✗            | ✗           | 0.0132     | 0.0247     |
| 仅场景             | ✗            | ✓           | 0.0196     | 0.0325     |
| **仅社交**         | ✓            | ✗           | **0.0118** | 0.0210     |
| 完整 (场景 + 社交) | ✓            | ✓           | 0.0137     | **0.0184** |

仅社交变体实现了**最佳 ADE**（0.0118），比基线降低了 10.6%，证明行人间交互是平均轨迹精度的主导因素。有趣的是，仅场景变体在两项指标上都出现了*退化*——可能是因为在没有社交上下文的情况下，场景编码器引入了模型无法消歧的噪声。完整模型通过结合两种特征流进行了补偿，实现了**最佳 FDE**（0.0184），这表明场景信息在轨迹终点最有帮助，因为在该处环境约束（可步行区域、障碍物）变得具有决定性。

来源: [ablation_results.md](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/references/ablation_results.md#L1-L14)

## 配置参考

SocialGAT 模块通过模型级开关和数据级邻居参数进行控制：

| 配置键                 | 文件           | 默认值 | 对 SocialGAT 的影响                    |
| ---------------------- | -------------- | ------ | -------------------------------------- |
| `model.use_social`     | `default.yaml` | `true` | 开启/关闭整个社交模块                  |
| `model.hidden_dim`     | `default.yaml` | `64`   | 控制 Agent/邻居特征维度及 `fuse` 输出  |
| `data.neighbor_radius` | `default.yaml` | `2.0`  | 包含邻居的空间阈值（米）               |
| `data.max_neighbors`   | `default.yaml` | `16`   | 每个 Agent 的最大邻居数量；设置 K 维度 |

`SocialGAT` 的 `attn_dim` 参数在 `CSGATNet` 构造函数调用中被硬编码为 `32`。要更改它，你需要修改 [csgat_net.py:37](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L37) 处的实例化代码。

来源: [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L11-L23), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L37)

## 接下来是什么

既然你已经了解了如何通过掩码图注意力捕捉社交交互，架构中自然的下一步是：

- **[ResNet 场景语义编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/6-resnet-scene-semantic-encoder)** — 第二个特征流，从场景图像中编码可步行区域的语义，当与社交特征结合时能产生最佳的 FDE。
- **[特征融合与自回归解码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/8-feature-fusion-and-autoregressive-decoder)** — 社交特征与 Agent 和场景特征拼接之处，被投影为条件向量，并输入到 CVAE + LSTM 解码器中以生成轨迹。
- **[轨迹数据集与邻居构建](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/10-trajectory-dataset-and-neighbor-construction)** — 深入了解 `neighbor_radius` 和 `max_neighbors` 如何塑造 SocialGAT 处理的数据。

# ResNet 场景语义编码器

ResNet 场景语义编码器将原始场景图像——无论是语义分割图、RGB 俯视照片，还是预计算的特征数组——转换为捕捉行人运动**物理环境约束**的紧凑向量。在 CSGAT-Net 流程中，该向量会与轨迹特征和社交特征拼接，随后融合为 CVAE 头的统一条件表示。对于希望使用更丰富的场景输入扩展模型、替换主干网络，或分析消融实验所揭示的经验权衡的开发者而言，理解该模块至关重要。

## 为什么场景特征很重要

行人轨迹并非仅受社交互动支配。在 ETH 校园行走的人会沿着步道行走、避开种植箱、绕过建筑拐角——所有这些都是纯轨迹模型无法感知的**静态环境约束**。场景编码器通过将局部环境的图像投影到隐空间来解决这一盲点，在该隐空间中，可通行区域、障碍物和空间布局被隐式编码。当场景特征向量与轨迹嵌入融合时，下游解码器便能获取关于*运动在物理上何处合理*的信息，而不仅仅是*相似智能体过去如何运动*。

该设计遵循项目参考资料中记载的“CNN + 直接特征提取”策略——有意在推理时避免进行完整的语义分割以降低计算成本，同时仍能捕捉场景的基本空间结构。

来源: [references/_extracted_scene_features.txt](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/references/_extracted_scene_features.txt#L1-L50), [references/_extracted_scene_features.txt](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/references/_extracted_scene_features.txt#L143-L147)

## 架构概述

`models/resnet_seg.py` 中的 `SceneEncoder` 类是一个三阶段的流程：**主干特征提取 → 全局池化 → 线性投影**。主干网络是一个截断的 ResNet（标准 torchvision ResNet 的第 0 层至第 N-2 层，即除自适应平均池化和全连接分类头之外的*所有部分*）。这保留了最后一个卷积块的空间特征图，随后通过 `AdaptiveAvgPool2d((1,1))` 将其压缩为每个通道的单一描述符，最后投影到所需的输出维度。



关键的架构决策如下：

| 决策                     | 实现                                           | 理由                                                         |
| ------------------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| 移除最后两个 ResNet 阶段 | `nn.Sequential(*list(resnet.children())[:-2])` | 丢弃专用于分类的池化层和全连接层，保留用于场景编码的空间特征图 |
| 自适应平均池化           | `nn.AdaptiveAvgPool2d((1,1))`                  | 无论输入分辨率如何，都能聚合空间信息——该编码器与分辨率无关   |
| 线性投影                 | `nn.Linear(resnet.fc.in_features, out_dim)`    | 将高维主干输出（ResNet18 为 512 维，ResNet34 为 512 维）映射到可配置的 `scene_dim` |
| 条件性主干选择           | 通过字符串参数选择 `resnet18` / `resnet34`     | 平衡模型容量与推理速度；ResNet18 是默认的轻量级选择          |

来源: [models/resnet_seg.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/resnet_seg.py#L9-L44)

## 构造函数参数



```
SceneEncoder(    in_channels: int = 3,          out_dim: int = 128,        # 输出特征维度（映射到配置中的 scene_dim）    backbone: str = "resnet18", # 主干架构："resnet18" 或 "resnet34"    pretrained: bool = False,   # 是否加载 ImageNet 预训练权重)
```

**`in_channels`** — 控制第一个卷积层。当设置为 3（默认值）时，标准 ResNet 的 `conv1` 保持不变。对于非 RGB 输入——例如单通道语义标签图或多通道特征堆栈——编码器会自动将 `conv1` 替换为新的 `nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)` 以匹配通道数。请注意，此替换会丢弃第一层的所有预训练权重。

**`out_dim`** — 输出场景特征向量的维度。在默认配置中，该值为 `scene_dim: 128`。此值直接影响 `CSGATNet` 中融合层的输入大小：当场景模块和社交模块同时启用时，总融合维度为 `hidden_dim + scene_dim + hidden_dim`。

**`backbone`** — 仅支持 `"resnet18"` 和 `"resnet34"`。两者在投影层前均生成 512 维的特征向量。传入其他任何字符串将引发 `ValueError`。

**`pretrained`** — 当为 `True` 时，加载对应的 `torchvision` ImageNet 预训练权重。当场景输入为自然 RGB 图像时，这可以加速收敛；但当输入为统计数据与 ImageNet 差异显著的合成分割图时，则可能适得其反。





使用单通道语义标签图（整数类 ID）时，你必须设置 `in_channels=1`。然而，原始整数标签并非理想的 CNN 输入——在将其输入编码器之前，建议将标签归一化至 [0,1] 区间，或将其转换为独热编码 / 嵌入表示。



来源: [models/resnet_seg.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/resnet_seg.py#L10-L39), [configs/default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L17-L23)

## 前向传播数据流

前向传播方法被刻意保持极简——按顺序执行三个操作：



```
def forward(self, x: torch.Tensor) -> torch.Tensor:    feat = self.backbone(x)          # (B, 512, H', W')    pooled = self.pool(feat).flatten(1)  # (B, 512)    return self.fc(pooled)           # (B, out_dim)
```

**输入张量形状**：`(B, C, H, W)`，其中 `C` 必须与 `in_channels` 匹配。空间维度 `H, W` 是灵活的，因为自适应池化会在内部将其归一化为 `(1, 1)`。本项目中的典型场景图以 NumPy 数组形式加载，并通过数据集的 `__getitem__` 方法转换为 `(C, H, W)` 张量。

**输出张量形状**：`(B, out_dim)` — 一个逐样本的场景描述符，后续将与轨迹特征 `(B, hidden_dim)` 以及可选的社交特征 `(B, hidden_dim)` 拼接。

计算开销主要由主干网络的前向传播决定。对于 ResNet18，处理每个 224×224 的 RGB 输入约需 ~1.8 GFLOPs——这足以支持实时推理，但在使用大批量训练时开销依然显著。

来源: [models/resnet_seg.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/resnet_seg.py#L41-L44), [data/dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L186-L189)

## 场景图加载流程

场景编码器本身不加载数据——该职责属于数据流水线。理解场景图如何传递到编码器对于调试和扩展至关重要。

加载链的工作流程如下：



`data/dataset.py` 中的 `_load_scene_map` 函数支持两种输入模态：

| 格式       | 扩展名            | 加载方法                          | 输出形状                |
| ---------- | ----------------- | --------------------------------- | ----------------------- |
| NumPy 数组 | `.npy` 或 `.npz`  | `np.load()`                       | `(H, W)` 或 `(H, W, C)` |
| 栅格图像   | `.png`、`.jpg` 等 | `PIL.Image.open().convert("RGB")` | `(H, W, 3)`             |

当 `scene_dir` 为 `null`（所有提供的配置中的默认值）或给定 `scene_id` 的场景文件不存在时，`_load_scene_map` 将返回 `None`。随后，数据集会将 `scene_t` 设为 `None`，而 `collate_fn` 会将批次中的所有 `None` 条目替换为具有相同形状的零张量。这种**优雅降级**机制确保了模型即使在场景数据部分可用时也能进行训练。





在当前配置中，`scene_dir` 被设置为 `null`，这意味着场景编码器在训练期间接收的是全零张量。要激活真实的场景特征，请创建一个包含按场景命名的 `{scene_id}.npy` 文件（如 `seq_eth.npy`、`seq_hotel.npy`）的目录，并在配置中将 `scene_dir` 指向该目录。



来源: [data/dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L22-L38), [data/dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L148-L194), [data/dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L198-L214), [configs/default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L14-L15)

## 与 CSGATNet 的集成

`SceneEncoder` 根据 `use_scene` 标志在 `CSGATNet.__init__` 内部进行条件性实例化：



```
self.scene_encoder = SceneEncoder(in_channels=3, out_dim=scene_dim) if use_scene else None
```

在前向传播期间，场景特征被生成并拼接到多源特征列表中：



```
feats = [agent_feat]                    # (B, hidden_dim) — 来自 LSTM 轨迹编码器if self.use_scene:    if scene_map is None:        scene_feat = torch.zeros(...)   # 无场景数据时的零向量回退    else:        scene_feat = self.scene_encoder(scene_map)  # (B, scene_dim)    feats.append(scene_feat)if self.use_social:    feats.append(social_feat)           # (B, hidden_dim) — 来自 SocialGAT cond_feat = self.fuse(torch.cat(feats, dim=-1))  # Linear + ReLU → (B, hidden_dim)
```

融合维度是动态计算的：

| 配置                                         | 融合输入维度                                          | 融合层                   |
| -------------------------------------------- | ----------------------------------------------------- | ------------------------ |
| 基准 (`use_scene=False, use_social=False`)   | `hidden_dim` (64)                                     | `Linear(64, 64) + ReLU`  |
| 仅场景 (`use_scene=True, use_social=False`)  | `hidden_dim + scene_dim` (64+128=192)                 | `Linear(192, 64) + ReLU` |
| 仅社交 (`use_scene=False, use_social=True`)  | `hidden_dim + hidden_dim` (64+64=128)                 | `Linear(128, 64) + ReLU` |
| 完整模型 (`use_scene=True, use_social=True`) | `hidden_dim + scene_dim + hidden_dim` (64+128+64=256) | `Linear(256, 64) + ReLU` |

`self.fuse` 层在拼接的异构特征进入 CVAE 条件路径之前，将其映射回 `hidden_dim`。这种**基于拼接的融合**是组合轨迹和场景嵌入的最简单策略——虽然有效且稳定，但可能不如基于注意力的方法那样能很好地捕捉细粒度的跨模态交互。

当推理时 `scene_map` 为 `None` 但 `use_scene=True`，模型会回退到形状为 `(B, scene_dim)` 的零向量。此设计确保了使用场景特征训练的模型在场景数据不可用时仍能生成预测——只是输出将缺乏环境约束依据。

来源: [models/csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L36-L46), [models/csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L85-L110)

## 配置参考

场景编码器由两个配置部分控制：

**数据级设置**（位于 YAML 的 `data:` 下）：

| 键          | 默认值 | 描述                                                    |
| ----------- | ------ | ------------------------------------------------------- |
| `scene_dir` | `null` | 包含逐场景地图文件的目录路径。`null` 表示禁用场景加载。 |
| `scene_ext` | `.npy` | 场景图文件的扩展名。支持 `.npy`、`.npz` 及图像格式。    |

**模型级设置**（位于 YAML 的 `model:` 下）：

| 键          | 默认值 | 描述                                                         |
| ----------- | ------ | ------------------------------------------------------------ |
| `scene_dim` | `128`  | 场景编码器的输出维度；映射到 `SceneEncoder` 中的 `out_dim`。 |
| `use_scene` | `true` | 主开关。当为 `false` 时，不实例化 `SceneEncoder`，且融合时排除场景特征。 |

要启用场景编码，请按如下方式更新配置：



```
data:  scene_dir: data/scene_maps    # 指向你的场景图目录  scene_ext: .npy               # 对于基于图像的地图使用 .png model:  scene_dim: 128                # 保持默认或调整  use_scene: true               # 启用场景编码器
```

消融实验配置演示了不同的操作模式：[ablation_baseline.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_baseline.yaml) 设置 `use_scene: false`，而 [ablation_scene.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_scene.yaml) 设置 `use_scene: true` 以及 `use_social: false` 以单独验证场景的贡献。

来源: [configs/default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L14-L23), [configs/ablation_scene.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_scene.yaml#L17-L23), [configs/ablation_baseline.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_baseline.yaml#L17-L23)

## 消融分析：场景特征的影响

项目的消融研究量化了场景特征的贡献：

| 模型                    | ADE ↓  | FDE ↓  | 相对基准的 Δ ADE    | 相对基准的 Δ FDE |
| ----------------------- | ------ | ------ | ------------------- | ---------------- |
| 基准（仅 LSTM）         | 0.0132 | 0.0247 | —                   | —                |
| 仅场景                  | 0.0196 | 0.0325 | +0.0064（变差）     | +0.0078（变差）  |
| 仅社交                  | 0.0118 | 0.0210 | −0.0014（变好）     | −0.0037（变好）  |
| 完整模型（场景 + 社交） | 0.0137 | 0.0184 | +0.0005（略微变差） | −0.0063（变好）  |

对这些结果进行细致的解读至关重要。**仅场景**模型在两个指标上的表现均逊于基准，这似乎暗示场景特征并无帮助。然而，两个关键因素解释了这一点：(1) 所有提供的配置均设置 `scene_dir: null`，这意味着场景编码器在这些实验中很可能接收了全零输入，使其沦为无效的参数沉没区；(2) 场景特征在为社交特征提供**互补**信息时最具价值——完整模型取得了最佳的 FDE (0.0184)，相比基准提升了 25.5%，这正是因为场景约束优化了终点预测，即使它们并未改善平均位移。

这与参考文献中的观察一致，即场景特征最有助于提升**终点精度**（此时障碍物和步道边界约束了最终位置），而社交特征则主导**路径级精度**（此时交互动态塑造了轨迹形状）。

来源: [references/ablation_results.md](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/references/ablation_results.md#L1-L14), [references/_extracted_scene_features.txt](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/references/_extracted_scene_features.txt#L96-L106)

## 扩展场景编码器

当前的实现刻意保持极简，使其易于扩展：

**替换主干网络** — 在构造函数中为任何 torchvision 模型（如 ResNet50、EfficientNet-B0）添加新的 `elif` 分支。切记更新 `resnet.fc.in_features` 以匹配新主干网络的最终特征维度。

**支持更丰富的输入** — `in_channels` 参数已能处理非 RGB 输入。对于具有 K 个类别的语义分割图，可将其转换为独热编码（K 个通道），或在输入主干网络前使用可学习的嵌入层。或者，直接用 SegFormer 等轻量级分割模型替换整个主干网络，并使用其编码器输出。

**增加空间感知** — 当前设计通过全局平均池化压缩了所有空间信息。若要保留空间结构，可将 `AdaptiveAvgPool2d` 替换为位置编码或空间注意力模块，并返回特征图 `(B, D, H', W')` 而非向量。这需要对 `CSGATNet` 中的融合层进行相应修改。

**预计算特征** — 如果你已拥有存储为 `.npy` 文件的预提取场景特征向量，则可以通过在数据集中加载它们，并作为形状为 `(D,)` 的 `scene_map` 直接传入，从而完全绕过主干网络。编码器将需要一个简化的前向路径（或者你可以设置 `use_scene=True` 并将编码器替换为恒等直通）。

来源: [models/resnet_seg.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/resnet_seg.py#L24-L39), [data/dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L22-L38)

## 下一步

既然你已了解场景编码器如何提取和投影环境特征，自然的下一步便是了解这些特征如何与 CVAE 采样机制交互以生成多模态预测。[CVAE 多模态轨迹头](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/7-cvae-multimodal-trajectory-head)页面涵盖了以融合的场景+轨迹+社交向量为条件的隐变量模型。或者，如果你想了解与场景特征并行拼接的另一特征流，请访问[社交图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network)页面。



# CVAE 多模态轨迹头

**CVAE（条件变分自编码器）头** 是 CSGAT-Net 的概率核心——该模块将确定性特征编码转换为**结构化潜在空间**，使模型能够生成多样且合理的未来轨迹。与将条件映射为单一预测的确定性解码器不同，CVAE 头对人类运动的**固有 multimodality（多模态性）**进行建模：在任何给定的观测下，行人可能会左转、直行或右转。通过在训练期间学习条件潜在分布，并在推理期间从先验中采样，该头生成 *K* 个轨迹假设，这些假设共同覆盖了合理未来的空间。

## 架构蓝图

CVAE 头实现了一个经典的**编码器-解码器潜在变量模型**，包含两个识别网络——一个**后验网络**（在训练期间由真实未来信息指导）和一个**先验网络**（在推理时仅以观测到的上下文为条件）。两个网络均输出均值-对数方差对，而重参数化技巧则弥合了采样与可微性之间的鸿沟。



后验网络接收 `cond_feat` 和 `fut_feat` 的拼接，使其能优先获取真实的未来信息，从而学习将潜在编码放置在真实发生的模式附近。先验网络仅接收 `cond_feat`，使其在无法获取未来信息的推理阶段依然可用。两个网络共享相同的两层 MLP 架构（`Linear → ReLU → Linear`），但输入维度和学习参数不同。

来源: [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L1-L43), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L46-L49)

## 类接口与参数约定

`CVAEHead` 类暴露了一个最小化的专用接口，包含四个构造函数参数和一个前向方法：

| 参数         | 类型  | 默认值 | 描述                                                     |
| ------------ | ----- | ------ | -------------------------------------------------------- |
| `cond_dim`   | `int` | —      | 融合条件向量的维度（在默认配置中等于 `hidden_dim` = 64） |
| `fut_dim`    | `int` | —      | 未来轨迹编码的维度（在默认配置中等于 `hidden_dim` = 64） |
| `latent_dim` | `int` | `16`   | 潜在空间 *z* 的维度；控制多模态表达能力                  |
| `hidden_dim` | `int` | `128`  | 先验和后验网络的内部 MLP 宽度                            |

前向方法签名及其返回元组在训练和推理模式下有所不同：



```
def forward(self, cond_feat, fut_feat=None) ->    (z, prior_mu, prior_logvar, post_mu, post_logvar)
```

| 模式     | `fut_feat` | z 来源   | `post_mu / post_logvar` |
| -------- | ---------- | -------- | ----------------------- |
| **训练** | 已提供     | 后验采样 | 返回 (非 None)          |
| **推理** | `None`     | 先验采样 | `None, None`            |

来源: [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L7-L42)

## 双分支识别网络

### 后验网络——训练时的先知

后验网络是一个两层 MLP，将拼接后的 `[cond_feat; fut_feat]` 映射为 `latent_dim × 2` 的向量，然后将其拆分为均值和对数方差：



```
[cond_feat ⊕ fut_feat]  →  Linear(cond_dim+fut_dim, hidden_dim)  →  ReLU  →  Linear(hidden_dim, latent_dim×2)  →  (μ_q, log σ²_q)
```

在训练期间，CSGATNet 通过 `fut_encoder`（一个展平 + 线性层 + ReLU）对真实未来轨迹进行编码以生成 `fut_feat` [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L48)。该未来编码在进入后验网络之前与 `cond_feat` 拼接，确保后验分布**锚定于实际发生的情况**。随后，来自该后验的重参数化样本 *z* 被馈入自回归解码器以生成训练预测。

### 先验网络——推理时的替代者

先验网络沿用了后验网络的架构，但仅接收 `cond_feat`：



```
[cond_feat]  →  Linear(cond_dim, hidden_dim)  →  ReLU  →  Linear(hidden_dim, latent_dim×2)  →  (μ_p, log σ²_p)
```

在推理时，先验必须在缺乏未来信息的情况下近似后验。后验与先验之间的 **KL 散度**损失作为训练信号，迫使先验成为一个良好的替代者——这两个分布越接近，KL 惩罚就越小，先验在推理期间就能更好地生成有意义的潜在编码。





CVAE 头的先验**并非**固定的标准正态分布 N(0, I)——它是一个由 `cond_feat` 参数化的**学习到的条件分布**。这是与普通 VAE 的关键区别：先验能够适应每个行人的上下文，生成比通用各向同性高斯分布信息量大得多的上下文感知潜在分布。



来源: [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L12-L21)

## 重参数化与采样机制

重参数化技巧作为一个静态方法实现，保留了随机采样时的梯度流：



```
@staticmethoddef sample(mu, logvar):    eps = torch.randn_like(mu)    return mu + eps * torch.exp(0.5 * logvar)
```

这将采样操作从 `z ~ N(μ, σ²)` 转换为 `(μ, σ)` 和辅助噪声 `ε ~ N(0, I)` 的确定性函数，允许通过采样步骤进行反向传播。`0.5 * logvar` 指数从对数方差中恢复标准差，而 `torch.randn_like` 提供了可微的噪声注入。

`_split` 方法使用 `chunk(2, dim=-1)` 沿特征维度将网络的 `latent_dim × 2` 输出拆分为均值和对数方差两部分——这是一种数值稳定的索引替代方案，因为它避免了可能干扰自动求导的原地操作。

来源: [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L23-L32)

## 训练与推理数据流

CVAE 头的行为在 `fut_feat` 门控处产生分岔。理解这种分流对于调试和扩展架构都至关重要。



**训练**通过从后验中抽取一个 *z*，为每个样本生成单一轨迹预测。返回的四元组 `(μ_p, log σ²_p, μ_q, log σ²_q)` 直接馈入 KL 散度损失。**推理**从先验中抽取 *K* 个独立样本，将每个样本解码为完整轨迹，并将它们堆叠为 `(B, K, T, 2)` 张量，用于 Best-of-K 评估。





在推理期间，*K* 个潜在样本中的每一个都使用**相同**的 `cond_feat`，但随机噪声 `ε` 不同，因此预测的多样性完全源于先验的方差结构。如果先验发生了坍缩（方差接近零），所有 *K* 个样本将几乎相同——这是**后验坍缩**的征兆，表明可能需要调整 KL 权重 `β`。



来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L112-L126), [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L34-L42)

## 与 CSGATNet 的集成

CVAE 头并非孤立运行——它作为 **特征融合层**和**自回归解码器**之间的桥接器接入 CSGATNet。关键的集成点如下：

1. **条件构造**——`fuse` 层将智能体的轨迹编码与可选的场景和社交特征拼接，然后投影到 `hidden_dim` 以形成 `cond_feat`。[csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L39-L46)
2. **未来编码**——`fut_encoder` 将真实未来轨迹 `(B, T_pred, 2)` 展平为 `(B, T_pred×2)` 并将其投影到 `hidden_dim`。此编码仅在训练期间使用。[csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L48)
3. **潜在变量注入解码器**——采样得到的 *z* 与 `cond_feat` 拼接，并通过 `dec_init` 传递，以生成 LSTM 解码器的初始隐藏状态 `(h0, c0)`，从而将生成过程锚定到上下文和潜在模式。[csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L51-L53)

| CSGATNet 组件      | CVAEHead 连接    | 维度流                                                       |
| ------------------ | ---------------- | ------------------------------------------------------------ |
| `fuse` 输出        | `cond_feat` 输入 | `fuse_dim` → `hidden_dim` (64) → CVAEHead                    |
| `fut_encoder` 输出 | `fut_feat` 输入  | `pred_len×2` (24) → `hidden_dim` (64) → CVAEHead             |
| CVAEHead `z` 输出  | `dec_init` 输入  | `hidden_dim + latent_dim` (80) → `hidden_dim×2` (128) → (h0, c0) |

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L39-L54), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L85-L126)

## 多模态推理：Best-of-K 采样

CVAE 头的实用价值在评估阶段通过 **Best-of-K** 采样得以体现。CSGATNet 中的推理循环从先验中抽取 `sample_k`（默认值：20）个独立潜在向量，将每个向量解码为完整的 12 步轨迹，并将它们堆叠成 `(B, K, T, 2)` 张量。随后，[ADE/FDE 指标](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/14-ade-fde-metrics-with-best-of-k)会为每个样本选择最佳假设——即与真实位移最小的假设。

这种评估协议直接奖励 CVAE 覆盖多种模式的行为：如果先验将其概率质量分布在多个合理方向上，则 *K* 个样本中至少有一个可能匹配实际结果。所有消融配置中默认的 `sample_k = 20` 在覆盖范围与计算成本之间取得了平衡。

| 配置                     | `latent_dim` | `beta` | `sample_k` | 备注                    |
| ------------------------ | ------------ | ------ | ---------- | ----------------------- |
| `default.yaml`           | 16           | 1.0    | 20         | 参考配置                |
| `ablation_baseline.yaml` | 16           | 1.0    | 20         | 仅轨迹消融              |
| `ablation_full.yaml`     | 16           | 1.0    | 20         | 完整模型（场景 + 社交） |
| `ablation_scene.yaml`    | 16           | 1.0    | 20         | 仅场景消融              |
| `ablation_social.yaml`   | 16           | 1.0    | 20         | 仅社交消融              |

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L119-L126), [configs/default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L17-L23), [configs/ablation_baseline.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_baseline.yaml#L17-L23)

## 损失耦合：KL 散度与重构

CVAE 头的潜在分布通过 [CVAE 损失函数](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/13-cvae-loss-function)进行联合优化，该函数平衡了两个目标：

- **重构损失**（MSE）——强制解码器在后验样本 *z* 的条件下生成接近真实轨迹的预测。
- **KL 散度**——惩罚后验 *q(z|x, y)* 与先验 *p(z|x)* 之间的差距，对潜在空间进行正则化，使得先验在推理时依然有效。

KL 散度被计算为学习到的先验和后验之间的完整高斯 KL 散度（而非相对于标准正态分布），这是 CVAE 设计中两个分布均被参数化的直接结果：

DKL(q∥p)=12∑d=1D(log⁡σp2σq2+σq2+(μq−μp)2σp2−1)DKL(q∥p)=21∑d=1D(logσq2σp2+σp2σq2+(μq−μp)2−1)

`beta` 系数（默认值 1.0）控制这种权衡：较高的值会强制更严格的后验-先验对齐（降低多模态性），而较低的值允许后验编码更多未来信息，但可能导致推理时的先验缺乏信息量。**β-VAE** 视角表明，调整此参数对于平衡多样性与准确性至关重要。

来源: [cvae_loss.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/losses/cvae_loss.py#L1-L33), [configs/default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L28-L31)

## 设计原理与扩展点

**为什么选择 CVAE 而不是普通 VAE？** 标准 VAE 使用固定的 N(0, I) 先验，要求潜在空间在没有上下文指导的情况下编码所有与轨迹相关的信息。CVAE 的条件先验 *p(z|x)* 在推理时的采样效率要高得多：它不再是盲目地从各向同性高斯分布中采样并期望某个样本能匹配上下文，而是已经将概率质量集中在上下文合理的模式周围。

**潜在维度**——默认的 `latent_dim = 16` 提供了足够的容量来表示多种轨迹模式，而不会过度正则化。增加此值允许更细粒度的多模态覆盖，但有后验坍缩的风险（KL 项占主导地位，并将两个分布推向无信息量的常数）。减小此值会限制表达能力，但会使训练更加稳定。

**潜在扩展**——当前架构使用独立的高斯后验/先验。研究方向包括：(1) 用于显式多模态表示的**高斯混合先验**；(2) 在潜在空间上应用**标准化流**以获得更丰富的分布；(3) **β 退火调度**，逐步增加 KL 权重以防止早期后验坍缩。

来源: [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L7-L10), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L14-L24)

------

**下一步**：CVAE 头的潜在样本 *z* 流入自回归解码器——有关完整的解码流程，请参阅 [特征融合与自回归解码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/8-feature-fusion-and-autoregressive-decoder)。有关训练潜在空间的损失机制，请参阅 [CVAE 损失函数](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/13-cvae-loss-function)。有关如何评估 K 个采样轨迹，请参阅 [Best-of-K 的 ADE/FDE 指标](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/14-ade-fde-metrics-with-best-of-k)。





# 特征融合与自回归解码器

CSGAT网络的预测能力依赖于两个紧密耦合的机制：**多流特征融合**，将异构空间线索压缩为单一的条件向量；以及**自回归LSTM解码器**，通过累积位移增量逐步展开未来轨迹。本页将详细剖析这两个子系统——从基于拼接的融合门及其维度计算，到CVAE隐变量桥接，再到将隐变量采样转化为具体坐标序列的逐步解码循环。

## 三流特征融合架构

融合子系统接收三个独立编码的特征向量，并将它们合并为统一的条件表示`cond_feat`。每个流均源自前文文档所述的专用编码器：

- **轨迹流** — 维度为`hidden_dim`（默认64）的`agent_feat`，由[轨迹LSTM编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/4-trajectory-lstm-encoder)处理Agent观测的8步轨迹生成。
- **场景流** — 维度为`scene_dim`（默认128）的`scene_feat`，由[ResNet场景语义编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/6-resnet-scene-semantic-encoder)处理鸟瞰语义图生成。
- **社交流** — 维度为`hidden_dim`（默认64）的`social_feat`，由[社交图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network)聚合邻居特征生成。

融合维度`fuse_dim`在构建时基于`use_scene`和`use_social`配置标志动态计算。基础维度始终为`hidden_dim`（轨迹），每个启用的流会追加其输出维度：

| 配置   | `use_scene` | `use_social` | `fuse_dim`      | 活跃流             |
| ------ | ----------- | ------------ | --------------- | ------------------ |
| 基线   | `false`     | `false`      | 64              | 仅轨迹             |
| 仅场景 | `true`      | `false`      | 192 (64+128)    | 轨迹 + 场景        |
| 仅社交 | `false`     | `true`       | 128 (64+64)     | 轨迹 + 社交        |
| 完整   | `true`      | `true`       | 256 (64+128+64) | 轨迹 + 场景 + 社交 |

融合层本身是一个极简的两层MLP——`nn.Linear(fuse_dim, hidden_dim)`后接`nn.ReLU`——将拼接后的向量重新投影回规范的`hidden_dim`空间。这种设计确保了下游组件（CVAE头、解码器）无论启用了哪些流，始终接收相同尺寸的输入，从而使整个流水线高度模块化且便于消融实验。







融合层重新投影至`hidden_dim`的操作，使得在接口边界处进行消融切换的代价降为零。当禁用某个流时，其编码器被设置为`None`（不实例化），因此不会浪费任何参数。在运行时，被禁用的流仅从拼接列表中省略其特征，融合层的输入维度会在构建期间自动调整。



在前向传播过程中，融合操作以迭代方式组装。`feats`列表以`agent_feat`作为初始值，然后根据条件扩展`scene_feat`和`social_feat`。一个关键的防御性模式处理了推理时期望存在场景特征（`use_scene=true`）但未提供场景图的情况——此时会替换一个形状为`(batch, scene_dim)`的零向量，从而优雅地退化为仅轨迹条件而不会引发崩溃。

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L39-L46), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L85-L110), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L17-L23)

## CVAE隐变量桥接：从条件化到随机采样

融合后的`cond_feat`向量承担双重职责：它既作为CVAE先验和后验分布的条件，又用于初始化解码器的隐藏状态。[CVAE多模态轨迹头](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/7-cvae-multimodal-trajectory-head)负责生成隐向量`z`，将多模态随机性注入预测中。然而，融合与CVAE之间的交互值得在此深入探讨，因为`cond_feat`是融合后的场景/社交知识传递至解码器的唯一信息通道。

此阶段的数据流根据是否存在真实的未来轨迹分为两条路径：

**训练路径** — 未来轨迹被展平，并通过`fut_encoder`（一个`Linear(pred_len * 2, hidden_dim) → ReLU`序列）进行投影，以生成`fut_feat`。随后CVAE同时接收`cond_feat`和`fut_feat`，利用它们对后验分布`q(z | cond, fut)`进行参数化。通过重参数化从该后验分布中抽取隐变量`z`，确保梯度能够穿过随机节点。这使得解码器在训练期间能够获取蕴含未来信息的信号，同时损失函数中的KL散度项（在后验与先验之间计算）对隐空间进行正则化，使其仅凭`cond_feat`即可保持信息量。

**推理路径** — 当`fut_traj=None`时，仅对先验`p(z | cond)`进行参数化，并从中采样`z`。模型循环`sample_k`次（默认20次），从相同的`cond_feat`但不同的`z`采样中生成多个轨迹假设，从而启用[使用Best-of-K的ADE/FDE指标](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/14-ade-fde-metrics-with-best-of-k)中所使用的Best-of-K评估策略。

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L48-L49), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L112-L126), [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L34-L42)

## 自回归解码器：从隐变量到轨迹

解码器负责将抽象的条件信息（`cond_feat` + `z`）转化为具体的预测坐标序列。它采用了在`_decode`方法中实现的**位移增量**自回归设计，这在架构上与更为常见的直接坐标预测方法截然不同。

### 初始化：构建隐藏状态

解码器的LSTM隐藏状态由融合条件和隐变量采样确定性推导而来。`dec_init`层（`nn.Linear(hidden_dim + latent_dim, hidden_dim * 2)`）将拼接向量`[cond_feat; z]`（维度64 + 16 = 80）映射为大小为`hidden_dim * 2`（128）的向量，然后通过`chunk(2, dim=-1)`将其拆分为`h0`和`c0`。每一半都被重塑为PyTorch LSTM所期望的`(1, batch, hidden_dim)`格式：



```
init = dec_init(cat([cond_feat, z], dim=-1))   h0, c0 = init.chunk(2, dim=-1)                 # each (batch, 64)
h = h0.unsqueeze(0).contiguous()                # (1, batch, 64)
c = c0.unsqueeze(0).contiguous()                # (1, batch, 64)
```

这种设计意味着解码器从一个完整编码了Agent运动历史、场景上下文、社交交互以及多模态意图的状态开始——所有信息都被压缩进初始细胞状态。融合层输出的质量直接决定了解码器的起始点。

### 逐步解码循环

解码器逐步执行`pred_len`（默认12）个时间步，每一步生成一个坐标位移，并将其累加至上一个位置：



在每一步`t`中：

1. 上一个坐标`prev`（初始化为`last_obs = obs_traj[:, -1]`）通过`dec_input`（`nn.Linear(2, hidden_dim)`）进行投影，并扩展维度至形状`(batch, 1, hidden_dim)`。
2. 单步LSTM调用`decoder(inp, (h, c))`产生输出及更新后的隐藏状态。
3. 输出被压缩回`(batch, hidden_dim)`，并通过`out_fc`（`nn.Linear(hidden_dim, 2)`）映射为2D位移`delta`。
4. 预测坐标计算为`pred = prev + delta`（坐标级别的残差连接）。
5. `prev`被更新为`pred`，用于下一次迭代。

最终输出为`torch.stack(preds, dim=1)`，形状为`(batch, pred_len, 2)`。





位移增量公式`pred = prev + delta`对训练稳定性至关重要。直接坐标预测要求LSTM输出绝对位置，而随着预测时域的延伸，这些位置会越来越偏离训练分布。通过预测增量，每一步的目标都是一个小位移——这种分布在各个时间步中大致保持平稳，从而使优化景观显著平滑。这类似于残差连接稳定深度网络的方式，但应用于序列模型的输出层。



来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L51-L54), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L67-L83)

## 多模态推理：采样与Best-of-K

在推理期间，解码器在单次前向传播内被多次调用。`forward()`中的外层循环从先验分布中采样`sample_k`个独立的隐向量，分别进行解码，并堆叠所得预测：



```
preds = []for _ in range(sample_k):    z, prior_mu, prior_logvar, _, _ = self.cvae(cond_feat, None)    preds.append(self._decode(obs_traj[:, -1], cond_feat, z))pred = torch.stack(preds, dim=1)   # (batch, sample_k, pred_len, 2)
```

生成的张量具有一个额外维度，用于容纳`sample_k`个假设。[使用Best-of-K的ADE/FDE指标](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/14-ade-fde-metrics-with-best-of-k)中的评估流水线会为每个样本选择最接近真实值的假设，这是多模态轨迹预测评估的标准协议。

数值`sample_k`可通过YAML配置中的`eval.sample_k`进行配置（默认20），如[配置与消融设置](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/15-configuration-and-ablation-setup)所示。较高的值能提高覆盖真实轨迹的概率，但会线性增加推理成本。

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L119-L126), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L38-L40)

## 维度流概述

下表追踪了在完整模型配置（`use_scene=true`, `use_social=true`）下，批量大小为`B`的数据流经完整融合与解码流水线的张量维度：

| 阶段            | 操作                                        | 输入形状                 | 输出形状                  |
| --------------- | ------------------------------------------- | ------------------------ | ------------------------- |
| 轨迹编码        | `traj_encoder(obs_traj)`                    | `(B, 8, 2)`              | `(B, 64)`                 |
| 场景编码        | `scene_encoder(scene_map)`                  | `(B, 3, H, W)`           | `(B, 128)`                |
| 社交编码        | `social_gat(agent_feat, neigh_feats, mask)` | `(B, 64)` + `(B, N, 64)` | `(B, 64)`                 |
| 拼接            | `torch.cat(feats, dim=-1)`                  | 三个向量                 | `(B, 256)`                |
| 融合投影        | `fuse(cat_vec)`                             | `(B, 256)`               | `(B, 64)`                 |
| 未来编码 (训练) | `fut_encoder(fut_traj)`                     | `(B, 12, 2)` → 展平      | `(B, 64)`                 |
| CVAE先验        | `prior(cond_feat)`                          | `(B, 64)`                | `(B, 16)` × 2 (μ, logvar) |
| CVAE后验 (训练) | `posterior(cat([cond, fut]))`               | `(B, 128)`               | `(B, 16)` × 2 (μ, logvar) |
| 重参数化        | `z = μ + ε·exp(0.5·logvar)`                 | `(B, 16)`                | `(B, 16)`                 |
| 解码器初始化    | `dec_init(cat([cond, z]))`                  | `(B, 80)`                | `(B, 128)` → (h₀, c₀)     |
| 解码步骤        | `out_fc(LSTM(dec_input(prev)))`             | `(B, 2)`                 | `(B, 2)` (delta)          |
| 累加            | `pred = prev + delta`                       | `(B, 2)` + `(B, 2)`      | `(B, 2)`                  |
| 最终输出        | `torch.stack(preds, dim=1)`                 | 12 × `(B, 2)`            | `(B, 12, 2)`              |

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L14-L54), [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L7-L32)

## 消融对融合与解码的影响

消融配置直接控制哪些流参与融合，从根本上改变了`cond_feat`的信息内容，进而影响解码器的预测能力。根据实验结果：

| 模型               | `fuse_dim` | ADE ↓  | FDE ↓      |
| ------------------ | ---------- | ------ | ---------- |
| 基线 (仅LSTM)      | 64         | 0.0132 | 0.0247     |
| 仅场景             | 192        | 0.0196 | 0.0325     |
| 仅社交             | 128        | 0.0118 | 0.0210     |
| 完整 (场景 + 社交) | 256        | 0.0137 | **0.0184** |

一个反直觉的观察：与基线相比，仅添加场景特征（仅场景）反而*恶化*了ADE和FDE。这表明，在没有社交感知的情况下注入场景上下文时，融合层从192→64维度的投影可能由于压缩率增加而丢失了关键的轨迹信息。然而，当与社交特征结合时（完整模型），这三个流似乎提供了互补信息——尽管ADE较仅社交模型略有增加，但FDE降至所有配置中的最佳值，这表明场景和社交线索共同改善了长时域终点预测，即使短期步长精度受到了轻微影响。

融合投影瓶颈（无论`fuse_dim`如何，所有配置最终都收敛至`hidden_dim=64`）可能是仅场景配置性能下降的症结所在：与将64投影至64（基线）相比，通过单个线性层将192维投影至64维会丢弃更多每维信息。具有中间归一化的多层融合MLP可能可以缓解此问题，但目前单层设计优先考虑了简单性与可解释性。

来源: [ablation_results.md](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/references/ablation_results.md#L1-L14), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L40-L46)

## 接下来

你在此研究的融合与解码流水线，消费了由数据流水线预构建的邻居张量和场景图。要了解这些输入是如何构建的，请参阅[ETH/UCY数据集预处理](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/9-eth-ucy-dataset-preprocessing)和[轨迹数据集与邻居构建](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/10-trajectory-dataset-and-neighbor-construction)。关于端到端训练整个系统的损失函数，请参阅[CVAE损失函数](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/13-cvae-loss-function)。有关控制融合开关和采样参数的配置详细信息，请参阅[配置与消融设置](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/15-configuration-and-ablation-setup)。



# ETH/UCY 数据集预处理

ETH/UCY 基准是评估行人轨迹预测模型的事实标准，包含从两个大学校园的俯视摄像头捕获的五个不同场景。本文档说明了 `scripts/preprocess_eth_ucy.py` 如何接收 ETH 和 UCY 子数据集的异构原始标注格式，通过特定格式的解析器解决它们的结构差异，并将**统一的逐帧轨迹表示**（`frame pid x y`）输出到 `data/processed/` 目录中。理解此处理流程至关重要，因为下游的 [轨迹数据集与邻居构建](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/10-trajectory-dataset-and-neighbor-construction) 模块正需要这种标准化的布局。

来源：[preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L1-L185), [README.txt](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/ewap_dataset/README.txt#L1-L31)

## 原始数据格式及其差异

ETH 和 UCY 子数据集采用了截然不同的标注理念，预处理脚本必须对此进行协调。**ETH 场景**（`seq_eth`、`seq_hotel`）将密集的逐帧观测数据存储在 `obsmat.txt` 文件中，其中每行记录了行人在给定帧的世界坐标位置和速度。**UCY 场景**（Zara、Students、Arxiepiskopi）将稀疏的**样条控制点**存储在 `.vsp`/`.txt` 文件中，这意味着必须通过插值重建轨迹后才能使用。

| 属性       | ETH (`obsmat.txt`)                  | UCY (`.vsp` / `.txt`)  |
| ---------- | ----------------------------------- | ---------------------- |
| 标注风格   | 密集逐帧位置                        | 稀疏控制点（样条）     |
| 每行列数   | 8（frame, id, x, z, y, vx, vz, vy） | 可变块结构             |
| 坐标系     | 世界坐标（米，通过 H.txt）          | 像素 / 世界坐标        |
| 帧覆盖     | 存在所有标注帧                      | 仅关键帧；中间帧需插值 |
| 行内元数据 | 无                                  | `" - "` 分隔符后的注释 |
| 帧率       | 2.5 fps 标注（0.4秒步长）           | 可变；取决于场景       |

ETH `obsmat.txt` 格式将每次观测编码为八个以空格分隔的浮点值：`frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y`。`pos_z` 和 `v_z` 列（垂直于地平面）始终为零，在解析时会被丢弃。位置以米为单位表示，是使用存储在每个场景 `H.txt` 中的单应性矩阵从图像坐标投影而来的。

来源：[preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L19-L33), [README.txt](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/ewap_dataset/README.txt#L6-L12)

## 预处理流水线架构

预处理流水线采用双分支架构，最终汇聚为单一输出格式。下图展示了从原始标注到标准化训练/验证集划分的完整数据流：



每个分支执行特定格式的解析，然后由共享的 `_split_by_scene` 函数将结果划分到训练脚本所期望的训练和验证目录中。

来源：[preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L141-L184)

## ETH Obsmat 解析器 — 列提取

ETH 解析器 `_parse_eth_obsmat` 读取 `obsmat.txt` 文件的每一行，按空格分割，并仅提取四个相关列：**帧**（第 0 列）、**行人 ID**（第 1 列）、**x 坐标**（第 2 列）和 **y 坐标**（第 4 列）。z 坐标（第 3 列）和所有速度列（第 5-7 列）均被静默丢弃。帧和 ID 都通过 `int(float(...))` 转换为整数，以处理类似 `7.8000000e+02` 的科学记数法表示：



```
# 解析结果： (780, 1, 8.456844, 3.588066)frame = int(float(parts[0]))pid  = int(float(parts[1]))x    = float(parts[2])y    = float(parts[4])          # 跳过 parts[3] (z) 和 parts[5:8] (速度)
```

作为针对格式错误行的防御性措施，字段数少于 5 个的行将被跳过。生成的 `(frame, pid, x, y)` 元组列表直接存储到以序列名称为键的 `scenes` 字典中。

来源：[preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L20-L33)

## UCY 样条解析器 — 从控制点插值

UCY 解析器要复杂得多，因为 `.vsp` 标注文件将轨迹编码为**稀疏样条控制点**，而非逐帧位置。解析器必须通过在连续控制点之间进行线性插值来重建密集轨迹。解析过程分为三个阶段：

**阶段 1 — 注释和元数据剥离。**`_strip_inline_comment` 函数移除 `" - "` 分隔符之后的任何文本（UCY 标注中嵌入了人类可读的行内注释）。以 `"-"` 开头的行（文件格式中的节分隔符）也会被丢弃，同样被丢弃的还有特殊文件 `crowd_file_format.txt`，该文件仅用于说明格式而不包含轨迹数据。

**阶段 2 — 块结构解析。**清洗后的文件遵循递归结构：第一个值声明样条（轨迹）的数量；对于每个样条，头部给出控制点的数量；随后是相应数量的 `(x, y, frame)` 三元组。解析器按顺序读取此结构，在展平的行列表中推进索引：



```
<num_splines>
  <num_control_points>
    x1 y1 frame1
    x2 y2 frame2
    ...
```

**阶段 3 — 线性插值。**`_interpolate_points` 函数获取每个样条排序后的控制点，并使用线性插值填充连续关键帧之间的每个整数帧：对于帧 `f0` → `f1` 及其位置 `(x0, y0)` → `(x1, y1)`，帧 `f` 处的位置计算为 `x = x0 + (x1 - x0) * t`，其中 `t = (f - f0) / (f1 - f0)`。这会将稀疏标注转换为与 ETH 原生提供的相同逐帧粒度。





由样条控制点标注的 UCY 场景在插值后会产生均匀间距的帧索引（0, 1, 2, ...），而 ETH 场景则保留其原始的视频帧号（例如 780, 786, 792, ...）。下游数据集加载器通过提取连续的观测窗口（无论绝对帧号如何）来标准化这种差异。



来源：[preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L37-L109), [preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L112-L121)

## 训练/验证场景划分

在两个分支都生成其 `(frame, pid, x, y)` 元组后，`_split_by_scene` 函数会根据可配置的名称列表将每个场景路由到训练或验证目录。默认划分遵循 ETH/UCY 基准测试中广泛使用的**留一场景法**约定：

| 划分       | 场景                                                         | 文件数量 |
| ---------- | ------------------------------------------------------------ | -------- |
| **训练集** | `seq_eth`, `seq_hotel`, `crowds_zara01`, `crowds_zara02`, `crowds_zara03`, `students001` | 6        |
| **验证集** | `students003`, `arxiepiskopi1`, `uni_examples`               | 2        |





默认的验证集（`students003`、`arxiepiskopi1`、`uni_examples`）是有意保持较小的。若要在所有五个标准场景（ETH、Hotel、Univ、Zara1、Zara2）中进行严格的留一场景法评估，你应依次将 `--val_scenes` 设置为每个目标场景并重新运行预处理脚本，且每次均需重新训练模型。



该划分可通过 CLI 参数 `--train_scenes` 和 `--val_scenes` 进行配置，接受逗号分隔的场景名称。名称未出现在任一列表中的场景将被静默排除在输出之外。

来源：[preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L125-L139), [preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L141-L156)

## 输出格式 — 统一轨迹标准

`_write_rows` 函数将每个场景输出为纯文本文件，每行一个观测值，格式为 `{frame} {pid} {x:.6f} {y:.6f}`。坐标保留六位小数精度写入，保持了原始标注的亚厘米级精度。输出目录结构反映了训练/验证的划分：



```
data/processed/
├── train/
│   ├── seq_eth.txt          # 8909 行
│   ├── seq_hotel.txt
│   ├── crowds_zara01.txt    # 120 行
│   ├── crowds_zara02.txt
│   ├── crowds_zara03.txt
│   └── students001.txt      # 523 行
└── val/
    ├── students003.txt      # 1009 行
    └── uni_examples.txt     # 84 行
```

每个文件都是自包含的：`frame` 列提供时序排列，`pid` 列标识单个行人，`(x, y)` 给出世界坐标位置。文件不写入表头。如果父目录不存在则会自动创建。此格式由 [轨迹数据集与邻居构建](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/10-trajectory-dataset-and-neighbor-construction) 模块直接使用，该模块读取这些文件并将其分割为固定长度的观测/预测窗口。

来源：[preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L12-L16), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L1-L5)

## 运行预处理脚本

预处理脚本从项目根目录调用，需要两个指向原始数据目录的强制参数：



```
python scripts/preprocess_eth_ucy.py \    --eth_dir data/ewap_dataset \    --ucy_dir data/crowds/data \    --out_dir data/processed \    --train_scenes "seq_eth,seq_hotel,crowds_zara01,crowds_zara02,crowds_zara03,students001" \    --val_scenes "students003,arxiepiskopi1,uni_examples"
```

| 参数             | 必需 | 默认值                          | 描述                                                       |
| ---------------- | ---- | ------------------------------- | ---------------------------------------------------------- |
| `--eth_dir`      | ✅    | —                               | 包含 `seq_eth/` 和 `seq_hotel/` 的 `ewap_dataset` 目录路径 |
| `--ucy_dir`      | ✅    | —                               | 包含 `.vsp`/`.txt` 标注文件的 `crowds/data` 目录路径       |
| `--out_dir`      | ❌    | `data/processed`                | 标准化轨迹文件的输出目录                                   |
| `--train_scenes` | ❌    | `seq_eth,seq_hotel,...`         | 逗号分隔的训练场景名称                                     |
| `--val_scenes`   | ❌    | `students003,arxiepiskopi1,...` | 逗号分隔的验证场景名称                                     |

请注意，在运行脚本之前，必须将 UCY `.vsp` 标注文件放置在 `--ucy_dir` 目录中——仓库的 `data/crowds/data/` 目录仅包含视频和图像文件；标注文件本身需要从 UCY Crowds by Example 项目获取并解压至该目录。

来源：[preprocess_eth_ucy.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/preprocess_eth_ucy.py#L141-L184)

## 后续步骤

一旦预处理脚本填充了 `data/processed/`，流水线将继续执行 [轨迹数据集与邻居构建](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/10-trajectory-dataset-and-neighbor-construction) 模块，该模块读取这些统一的轨迹文件，并构建模型在训练期间使用的具有空间邻居图的固定长度滑动窗口样本。有关引用这些处理后路径的完整训练配置，请参阅 [配置与消融设置](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/15-configuration-and-ablation-setup)。



# 轨迹数据集与邻居构建

`TrajectoryDataset` 是核心的数据抽象，它弥合了原始预处理轨迹文本文件与模型预期张量输入之间的鸿沟。它作用于由 [ETH/UCY 数据集预处理](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/9-eth-ucy-dataset-preprocessing) 生成的标准化文件，并承担三项关键职责：**滑动窗口样本提取**、**空间邻居发现**和**坐标归一化**。所有下游模型组件——从 [轨迹 LSTM 编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/4-trajectory-lstm-encoder) 到 [社交图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network)——都使用该类生成的张量，这使得其构建逻辑成为预测质量在数据侧最重要的决定因素。

来源: [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L1-L10), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L81-L91)

------

## 样本数据结构

所有提取的样本在内部均由 **`Sample`** 数据类表示，它捕获了单个预测任务所需的全部信息：



```
@dataclassclass Sample:    obs: np.ndarray              fut: np.ndarray          # (pred_len, 2) 真实未来轨迹    neighbors: List[np.ndarray]  # 每个形状为 (obs_len, 2)，变长邻居列表    scene_id: str            # 场景名称，用于加载场景特征    agent_id: int            # 行人 ID，用于评估追踪
```

`obs` 和 `fut` 字段存储以米为单位的世界坐标系下的 2D 位置，其中 `obs` 涵盖 **8 帧观测窗口**，`fut` 涵盖 **12 帧预测时域**（可通过 `obs_len` / `pred_len` 配置）。`neighbors` 列表保存附近行人的观测轨迹——这种变长结构是有意为之的，因为人群密度在不同场景和时间步上差异极大。`scene_id` 字符串将每个样本映射到其对应的场景语义特征文件，而 `agent_id` 则保留了原始行人标识符，用于计算单体评估指标。

来源: [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L12-L18)

------

## 滑动窗口样本提取

函数 `_build_samples` 使用**滑动窗口**方法，将原始轨迹文本文件（格式：每行 `frame pedestrian_id x y`）转换为 `Sample` 对象列表。这是将连续帧流离散化为监督学习样本的过程。



算法首先按帧编号将所有行分组为 `frame_data`，然后以 `skip` 为步长遍历帧数组。在每个起始索引处，它将 `obs_len + pred_len` 个连续帧拼接成一个窗口。在此窗口内，它会识别在**每一帧**中都出现的行人——任何覆盖不完整（窗口内缺失帧）的行人都会被剔除。这种“完整覆盖”要求确保观测数据和真实未来轨迹均可用，无需进行插补。

| 参数       | 默认值 | 作用                                   |
| ---------- | ------ | -------------------------------------- |
| `obs_len`  | 8      | 观测帧数                               |
| `pred_len` | 12     | 预测帧数                               |
| `skip`     | 1      | 滑动窗口步长（1 = 包含所有可能的窗口） |
| `min_ped`  | 1      | 保留窗口所需的最少完整覆盖行人数       |

当 `skip=1` 时，连续的训练样本会有 `obs_len + pred_len - 1` 帧的重叠，从而生成密集的样本集。增大 `skip` 会以牺牲样本多样性为代价，减小数据集规模和训练时间。设置 `min_ped > 1` 会过滤掉仅有孤立行人的窗口，这在训练至少需要一名邻居才有意义的社交互动模型时非常有用。

来源: [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L48-L97)

------

## 邻居构建：空间半径过滤

邻居构建逻辑是 `_build_samples` 中架构意义最重要的部分，因为它直接决定了[社交图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network)的输入。其策略是**在最后一帧观测帧应用空间半径过滤**。

对于有效窗口中的每个目标行人，算法会遍历所有其他完整覆盖的行人，并计算他们在**帧 `obs_len - 1`**（最后一个观测时间步）位置之间的欧几里得距离。如果该距离在 `neighbor_radius` 范围内，则将另一位行人的**完整观测轨迹**（`obs_len` 帧）加入邻居列表：



```
dist = np.linalg.norm(ntraj[obs_len - 1] - obs[-1])if dist <= neighbor_radius:    neighbors.append(ntraj[:obs_len])
```

此逻辑中蕴含了两个关键的设计决策。首先，邻居资格由**单帧邻近度检查**决定，而非平均或累积距离——这捕获了预测开始那一刻的瞬时空间上下文，而这正是社会影响最关键的时刻。其次，邻居贡献的是其**整个观测轨迹**（而不仅仅是最后的位置），使下游 GAT 能够关注附近行人的时间模式，而不仅仅是他们的静态位置。

然后，邻居列表会被截断至 `max_neighbors` 项（默认 16），按发现邻居的顺序应用。这种固定容量的约束确保了批量计算时张量形状的一致性。

| 参数              | 默认值 | 作用                                 |
| ----------------- | ------ | ------------------------------------ |
| `neighbor_radius` | 2.0    | 最后观测帧处的最大欧几里得距离（米） |
| `max_neighbors`   | 16     | 每个样本的邻居数量上限（0 = 禁用）   |





默认的 `neighbor_radius=2.0` 米大致对应行人动力学文献中使用的“个人空间”半径（霍尔人际距离学：社交区域约 1.2–3.6 米）。此值捕获了足够近以至于能影响目标轨迹的行人，同时排除了远处的过客。调整此半径会直接影响社交互动特征的信噪比。



来源: [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L81-L95)

------

## `__getitem__` 中的填充与掩码

由于不同样本的实际邻居数量不同，`__getitem__` 方法会将邻居张量填充至**(max_neighbors, obs_len, 2) 的固定形状**，并生成一个形状为 `(max_neighbors,)` 的配套**二进制掩码**：



```
neigh_traj = np.zeros((self.max_neighbors, self.obs_len, 2), dtype=np.float32)neigh_mask = np.zeros((self.max_neighbors,), dtype=np.float32)for i, ntraj in enumerate(neighbors):    if i >= self.max_neighbors:        break    neigh_traj[i] = ntraj.astype(np.float32)    neigh_mask[i] = 1.0
```

掩码向量在下游至关重要：[社交图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network) 使用 `neigh_mask` 将填充位置的注意力对数值归零（在 softmax 前将其设为 `-1e9`），确保空槽位永远不会对聚合的社交特征产生贡献。如果没有此掩码，模型会将零填充视为一个站在坐标原点的行人并对其分配注意力——从而产生系统性偏差。

来源: [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L148-L161), [gat.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/gat.py#L14-L30)

------

## 坐标归一化：先居中后缩放

原始世界坐标在不同场景中跨越的范围不同（例如，ETH 序列使用约 5-15 米，而 UCY 序列可能使用不同的尺度）。数据集应用了**两步归一化**，通过将所有轨迹锚定到一个公共参考系来稳定训练：

1. **居中**：从所有坐标（目标观测、目标未来和邻居轨迹）中减去最后观测到的位置 `obs[-1]`。这使得最后观测点成为原点，因此模型预测的是**相对位移**而不是绝对位置。
2. **缩放**：将所有居中后的坐标除以 `obs`、`fut` 和 `neigh_traj` 拼接集合中的最大绝对值（限制最小为 `norm_eps`，默认 1e-6）。这将坐标范围压缩至大约 [-1, 1]，这正是神经网络激活函数最敏感的区间。



```
center = obs[-1].copy()obs = obs - centerfut = fut - centerneigh_traj = neigh_traj - center[None, None, :]max_val = np.max(np.abs(np.concatenate([obs, fut, neigh_traj.reshape(-1, 2)], axis=0)))scale = max(max_val, self.norm_eps)obs = obs / scalefut = fut / scaleneigh_traj = neigh_traj / scale
```

`center` 和 `scale` 会与归一化后的张量一并返回，以便在评估和可视化期间将预测结果**反归一化**回世界坐标。`collate_fn` 正是为了此目的，将它们堆叠成批级的 `center_t` 和 `scale_t` 张量。





缩放因子的计算包含了邻居轨迹，这意味着归一化范围会根据局部人群的空间范围进行自适应调整。在邻居距离较近的密集场景中，尺度会较小（保留细粒度的位移细节）；在稀疏场景中，尺度会相应扩大。这种自适应行为优于全局固定尺度，但也意味着相同的物理位移会根据上下文映射到不同的归一化值。



来源: [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L163-L192)

------

## 批次整理与张量接口

`collate_fn` 将各个样本组装成一个批量元组，处理某些样本的场景图可能为 `None` 的特殊情况（或者在未配置 `scene_dir` 时所有样本均为 `None`）。当批次中至少有一个样本带有场景图时，缺失的条目会用与第一个可用场景图形状匹配的**零张量**填充。

`__getitem__` 的返回签名和 `collate_fn` 的输出定义了训练循环和模型所使用的**数据契约**：

| 索引 | 张量       | 形状                             | 描述               |
| ---- | ---------- | -------------------------------- | ------------------ |
| 0    | `obs_t`    | `(B, obs_len, 2)`                | 归一化后的观测轨迹 |
| 1    | `fut_t`    | `(B, pred_len, 2)`               | 归一化后的未来轨迹 |
| 2    | `scene_t`  | `(B, C, H, W)` 或 `None`         | 场景语义图         |
| 3    | `neigh_t`  | `(B, max_neighbors, obs_len, 2)` | 填充后的邻居轨迹   |
| 4    | `mask_t`   | `(B, max_neighbors)`             | 二进制邻居掩码     |
| 5    | `idx_t`    | `(B,)`                           | 样本索引           |
| 6    | `agent_t`  | `(B,)`                           | 行人 ID            |
| 7    | `center_t` | `(B, 2)`                         | 反归一化中心       |
| 8    | `scale_t`  | `(B,)`                           | 反归一化缩放比例   |

训练循环在前向传播中直接使用此契约，格式为 `obs, fut, scene, neigh, mask, _, _, _, _`，其中下划线表示的字段（索引、agent_id、中心、缩放）用于评估和可视化，但不参与模型的前向计算。

来源: [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L197-L214), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L32-L37)

------

## 配置参考

所有数据集构建参数都集中在 YAML 配置文件中，并在实例化时传递给 `TrajectoryDataset`。下表记录了 data 部分的配置键及其对样本和邻居构建的影响：

| 键                     | 默认值                 | 影响范围 | 描述                           |
| ---------------------- | ---------------------- | -------- | ------------------------------ |
| `data.train_dir`       | `data/processed/train` | 文件加载 | 包含训练 `.txt` 文件的目录     |
| `data.val_dir`         | `data/processed/val`   | 文件加载 | 包含验证 `.txt` 文件的目录     |
| `data.obs_len`         | 8                      | 窗口切分 | 观测时域长度                   |
| `data.pred_len`        | 12                     | 窗口切分 | 预测时域长度                   |
| `data.skip`            | 1                      | 窗口切分 | 滑动窗口步长                   |
| `data.min_ped`         | 1                      | 窗口过滤 | 每个窗口中最少的完整覆盖行人数 |
| `data.neighbor_radius` | 2.0                    | 邻居构建 | 空间邻近阈值（米）             |
| `data.max_neighbors`   | 16                     | 邻居构建 | 每个样本的最大邻居数           |
| `data.scene_dir`       | `null`                 | 场景加载 | 预计算场景特征的目录           |
| `data.scene_ext`       | `.npy`                 | 场景加载 | 场景特征图的文件扩展名         |

所有四种消融实验配置（`baseline`、`social`、`scene`、`full`）共享**完全相同的数据参数**，确保性能差异仅归因于模型架构而非数据分布。`model` 部分中的 `use_scene` 和 `use_social` 标志控制模型是否处理这些特征，但数据集始终会构建并返回它们。

来源: [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L1-L15), [ablation_baseline.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_baseline.yaml#L1-L15)

------

## 端到端数据流

从预处理文本文件到模型可用张量的完整路径总结如下：



拆分目录中的每个 `.txt` 文件在 `__init__` 期间被读取一次，转换为 numpy 数组，并通过 `_build_samples` 生成 `Sample` 对象。随后，`__getitem__` 方法对每个单独的样本执行填充、归一化、可选增强（参见[数据增强变换](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/11-data-augmentation-transforms)）以及张量转换。`collate_fn` 通过堆叠张量和处理存在性可变的场景图来完成批次的组装。

来源: [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L100-L143), [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L148-L194), [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L197-L214)

------

## 下一步

在理解了数据集和邻居构建之后，以下页面将继续介绍数据管道并与模型端建立联系：

- **[数据增强变换](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/11-data-augmentation-transforms)** — `TrajectoryRotateScale` 如何将随机旋转和缩放应用于归一化轨迹
- **[社交图注意力网络](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/5-social-graph-attention-network)** — `neigh_t` 和 `mask_t` 如何被 GAT 消费以生成社交特征
- **[训练循环与检查点](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/12-training-loop-and-checkpointing)** — `collate_fn` 的输出如何被解包并输入模型



# 数据增强变换

数据增强是指通过对现有样本应用受控的随机变换，人为扩充训练集的实践。在行人轨迹预测中，数据增强发挥着至关重要的作用：像 ETH/UCY 这样的真实世界轨迹数据集规模相对较小（仅包含数千条序列），如果不进行增强，模型很容易记忆特定的运动模式，而非学习具备泛化能力的模式。本页将详解 **`TrajectoryRotateScale`** 变换——即本项目实现的唯一增强策略——以及它是如何接入数据集管线的。

## 为何增强轨迹？

行人轨迹具有一个实用的几何特性：一个以 1.2 m/s 的速度向东北行走的行人，与另一个以同样速度向西南行走的行人，表达了相同的*行为意图*——方向相对于世界坐标系是任意的，但*相对运动模式*却是完全一致的。通过在训练期间随机旋转轨迹，我们教导模型具备**旋转不变性**，防止其过拟合于特定场景坐标系的朝向。类似地，轻微的均匀缩放（例如 ±10%）有助于模型容忍步行速度和空间尺度的微小变化，提升对未知密度和环境的泛化能力。[transforms.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/transforms.py#L1-L9)

## TrajectoryRotateScale — 架构与数据流

`TrajectoryRotateScale` 类是一个**可调用变换**，遵循 `(obs, fut, neigh_traj, scene_map) → (obs, fut, neigh_traj, scene_map)` 的签名。这意味着它接收数据集产生的所有四个张量，但刻意保持 `scene_map` 不变——场景语义（例如，可通行区域、障碍物）属于绝对空间属性，绝不能对其进行旋转或缩放。



此处的核心洞见在于**随机门控机制**：以 `self.prob` 的概率应用该变换；否则，原始数据将原样通过。这确保了模型仍能以可控的比例看到未增强的样本，防止增强操作主导训练信号。[transforms.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/transforms.py#L21-L37)

## 旋转与缩放的数学原理

当该变换触发时，它会采样两个随机量：

| 参数    | 采样方法                      | 默认范围 | 效果                |
| ------- | ----------------------------- | -------- | ------------------- |
| `angle` | `np.random.uniform(-π, π)`    | 整圆     | 任意 2D 旋转        |
| `scale` | `np.random.uniform(0.9, 1.1)` | ±10%     | 轻微的空间膨胀/收缩 |

2×2 旋转矩阵采用经典构造方式：`rot = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]`。然后通过**右乘约定**对每条轨迹进行变换：`(traj @ rot.T) * scale`。这等效于对轨迹中的每个 2D 点依次应用旋转和各向同性缩放。[transforms.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/transforms.py#L25-L32)

**为何使用 `rot.T`？** 轨迹数组的形状为 `(T, 2)`——每一行代表一个点 `[x, y]`。矩阵乘法 `traj @ rot.T` 将每个输出点计算为 `[x·cos(θ) + y·sin(θ), -x·sin(θ) + y·cos(θ)]`，这正是标准的逆时针旋转角度 θ 的操作。对旋转矩阵使用 `.T`，使我们能够用一次简洁的 `@` 操作，代替转置后的左乘操作。

## 处理邻居张量的形状重塑

邻居轨迹以形状为 `(max_neighbors, obs_len, 2)` 的 **3D 张量**形式输入，但 `_apply` 函数期望的是一个 2D 数组，其中每一行是单个 `(x, y)` 点。该变换通过“重塑–应用–重塑”的模式处理此问题：



```
neigh_traj = _apply(neigh_traj.reshape(-1, 2)).reshape(neigh_traj.shape)
```

这会将所有邻居和时间步临时展平为一个 `(max_neighbors × obs_len, 2)` 矩阵，统一应用相同的旋转和缩放，然后恢复原始的 3D 结构。在单个样本中的所有邻居间使用**相同**的旋转矩阵和缩放因子，保留了目标行人与其邻居之间的**相对空间关系**——这是使社交图注意力机制保持语义有效性的关键要求。[transforms.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/transforms.py#L36)

## 场景地图 — 刻意排除

请注意，`scene_map` 是**未经修改**直接返回的：`return obs, fut, neigh_traj, scene_map`。这并非疏忽——而是源于几何推理的审慎设计决策。场景语义地图编码的是*绝对*空间布局：可行走路径、建筑边界和障碍物区域。独立旋转场景地图会导致**空间不一致**，即轨迹朝一个方向移动，而可行走区域却指向另一个方向。由于轨迹已经**以最后观测点为中心进行居中**（归一化在变换之前进行），且场景地图的坐标系是绝对的，旋转地图将需要进行相应的裁剪/变换，这远超简单的 2D 仿射变换范畴。安全且语义正确的做法是保持场景地图原样不变。[transforms.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/transforms.py#L37), [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L163-L179)

## 与数据集管线的集成

该变换作为**可选的归一化后步骤**，集成在 `TrajectoryDataset.__getitem__` 中。`__getitem__` 内部的执行顺序如下：



该变换作用于 **numpy 数组**（在第 181 行的 `torch.from_numpy` 转换之前），这意味着它在 CPU 上运行，并且兼容 PyTorch 的 `num_workers > 0` 多进程机制。变换接收的是已归一化的数据，因此旋转和缩放发生在**归一化坐标空间**中，其值通常在 `[-1, 1]` 范围内。[dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L148-L194)



归一化步骤将轨迹以最后观测点为中心居中，并按最大绝对值进行缩放。由于 `TrajectoryRotateScale` 在归一化*之后*运行，旋转操作是围绕原点（即 Agent 当前位置）进行的。这在几何上是正确的——你希望*相对于 Agent 当前位置*旋转未来轨迹，而不是围绕某个任意的世界空间原点旋转。

## 配置与默认行为

尽管变换类已完整实现，**默认的训练管线并未使用它**。检查 `train.py` 中 `TrajectoryDataset` 的实例化过程，并没有传递 `transform=` 参数，因此 `self.transform` 属性默认为 `None`，所有数据均以未增强的形式通过。这是一个审慎的实验选择——消融配置同样省略了变换，确保基线模型、仅社交模型、仅场景模型和完整模型之间的性能比较不会受到增强效应的干扰。[train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L81-L102)

要启用数据增强，你需要构造该变换并显式传递它：



```
from data.transforms import TrajectoryRotateScale transform = TrajectoryRotateScale(prob=0.5, rot_range=(-math.pi, math.pi), scale_range=(0.9, 1.1))train_ds = TrajectoryDataset(    split_dir=cfg["data"]["train_dir"],        transform=transform,)
```

请注意，该变换应**仅**应用于训练数据集，绝不能用于验证——在评估期间进行增强会给 ADE/FDE 指标引入不必要的方差。

## 参数参考

| 参数          | 类型                  | 默认值       | 描述                                                         |
| ------------- | --------------------- | ------------ | ------------------------------------------------------------ |
| `prob`        | `float`               | `0.5`        | 对每个样本应用变换的概率。较低的值能保留更多原始数据分布。   |
| `rot_range`   | `Tuple[float, float]` | `(-π, π)`    | 均匀随机旋转角度的范围（弧度）。整圆范围可确保完全的方向不变性。 |
| `scale_range` | `Tuple[float, float]` | `(0.9, 1.1)` | 均匀随机缩放因子的范围。小于 1.0 的值会收缩轨迹；大于 1.0 的值会膨胀轨迹。 |



如果你的模型迅速过拟合（训练损失快速下降，但验证 ADE 停滞或上升），可尝试设置 `prob=0.7` 和 `scale_range=(0.85, 1.15)` 以获得更强的正则化效果。如果你的数据集已经具备多样性（包含多个场景和不同的密度），请保持默认值或将 `prob` 降至 `0.3`，以避免扭曲罕见但重要的运动模式。

## 本变换未涵盖的内容

同样重要的是理解此处**未**实现哪些增强策略，因为每一项遗漏都代表着一个潜在的扩展点：

| 缺失的增强方式      | 为何重要                               | 实现复杂度                                     |
| ------------------- | -------------------------------------- | ---------------------------------------------- |
| **高斯噪声注入**    | 模拟真实世界追踪中的传感器测量噪声     | 低 — 添加 `np.random.normal(0, σ, traj.shape)` |
| **时间抖动**        | 将观测窗口偏移 ±1 帧，模拟时间未对齐   | 中 — 需要访问原始帧数据                        |
| **随机裁剪 / 遮挡** | 丢弃随机时间步或邻居，模拟部分观测     | 中 — 需要一致地更新掩码张量                    |
| **镜像/翻转**       | 水平翻转能以极低成本使方向覆盖范围翻倍 | 低 — 以 `prob` 概率对 x 坐标取反               |

上述每种策略都针对不同的失效模式：噪声注入针对传感器鲁棒性，时间抖动针对帧率敏感性，遮挡针对部分观测场景，镜像针对方向偏差。[transforms.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/transforms.py#L1-L38)

## 下一步去哪

你刚刚学习的变换是张量进入模型之前数据管线的最后阶段。为该变换提供数据的上游两个阶段记录在 [轨迹数据集与邻居构建](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/10-trajectory-dataset-and-neighbor-construction)（原始 `.txt` 文件如何转变为带有邻居列表的 `Sample` 对象）和 [ETH/UCY 数据集预处理](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/9-eth-ucy-dataset-preprocessing)（原始数据集文件如何转换为按场景划分的 `.txt` 格式）中。数据流出该变换后，便进入模型——从 [轨迹 LSTM 编码器](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/4-trajectory-lstm-encoder) 开始，将观测路径编码为隐状态。



# 训练循环与检查点

本项目的训练流程遵循经典的周期驱动循环，结合 CVAE 损失优化、周期性的 Best-of-K 验证以及双检查点持久化策略。该循环的各个环节——从超参数到模型拓扑——均由单一的 YAML 配置文件驱动，确保实验可通过单行命令**完全复现**。本文将介绍完整的训练生命周期：单个周期如何处理批次数据、验证如何选择最佳模型、检查点如何保存与加载，以及系统如何保证跨运行的确定性行为。

## 训练生命周期概述

整个训练过程可分解为四个按顺序执行的阶段，每个周期执行一次。下图展示了从数据加载到检查点持久化的完整循环。



入口点是 [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L70-L175) 中的 `main()` 函数，它解析 `--config` 参数（默认为 `configs/default.yaml`），加载配置字典，随后进行初始化并进入周期循环。将 `train_one_epoch` 和 `evaluate` 拆分为独立函数，使得 `main()` 中的控制流异常清晰：每个周期只需三行代码——训练、评估、有条件地保存检查点。

来源: [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L70-L175), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L1-L45)

## 单周期训练：前向传播、损失计算、反向传播

`train_one_epoch` 函数封装了标准的 PyTorch 训练步骤，对整个训练 DataLoader 进行单次遍历。它首先将模型设置为 `model.train()`，这对于 CVAE 头部**至关重要**——在训练期间，模型接收真实的未来轨迹（`fut`）并利用其计算后验分布，从而生成 KL 散度项所需的前验和后验统计量。



```
def train_one_epoch(model, loader, optimizer, device, beta):    model.train()    total_loss = 0.0    for obs, fut, scene, neigh, mask, _, _, _, _ in loader:        obs = obs.to(device)        fut = fut.to(device)        neigh = neigh.to(device)        mask = mask.to(device)        scene = scene.to(device) if scene is not None else None         pred, prior_mu, prior_logvar, post_mu, post_logvar = model(obs, fut, scene, neigh, mask)        loss = total_cvae_loss(pred, fut, prior_mu, prior_logvar, post_mu, post_logvar, beta)         optimizer.zero_grad()        loss.backward()        optimizer.step()         total_loss += loss.item()    return total_loss / max(1, len(loader))
```

DataLoader 通过 [collate_fn](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L198-L214) 生成一个 9 元组，训练循环使用了其中的前五个：`obs`（观测轨迹）、`fut`（未来真实值）、`scene`（可选的场景地图）、`neigh`（邻居轨迹）和 `mask`（邻居存在掩码）。末尾的四个值（索引、智能体 ID、中心点、缩放比例）预留给评估和可视化流程——它们携带了重建世界坐标预测所需的归一化元数据。

**Beta 加权 CVAE 损失**结合了重建项（预测轨迹与真实轨迹之间的 MSE）和由 `beta` 缩放的 KL 散度项。当 `beta=1.0`（默认值）时，这是标准的 VAE 目标函数；降低 beta 会削弱潜在空间的正则化，可能以结构化程度较低的潜在空间为代价来改善重建效果。损失在每个批次内求平均，随后在整个周期中累加，并除以批次数以得出该周期的平均训练损失。





模型的前向方法将 `fut_traj` 的存在与否作为隐式的训练/评估开关：当 `fut_traj is not None` 时，它将未来轨迹编码至后验分布；当 `fut_traj is None` 时，它从前验分布中采样。这就是为什么 `train_one_epoch` 传入 `fut`，而 `evaluate` 传入 `None` 的原因。



来源: [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L28-L47), [dataset.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/data/dataset.py#L198-L214), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L85-L126)

## Best-of-K 采样验证

在每个训练周期结束后，`evaluate` 函数使用 **Best-of-K 采样**在验证集上运行推理，这是多模态轨迹预测的标准评估协议。模型被设置为 `model.eval()`，从而禁用 Dropout 和 Batch Normalization 的随机性，并且推理循环将所有操作包裹在 `torch.no_grad()` 中以防止梯度累积。



```
def evaluate(model, loader, device, sample_k: int):    model.eval()    ade_list = []    fde_list = []    with torch.no_grad():        for obs, fut, scene, neigh, mask, _, _, _, _ in loader:            ...            pred, _, _, _, _ = model(obs, None, scene, neigh, mask, sample_k=sample_k)            ade_list.append(ade(pred, fut).item())            fde_list.append(fde(pred, fut).item())    return float(sum(ade_list) / max(1, len(ade_list))), float(sum(fde_list) / max(1, len(fde_list)))
```

当 `fut_traj=None` 时，[CSGATNet.forward](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L119-L126) 会循环 `sample_k` 次，每次从**前验分布**中提取一个潜在向量 `z` 并将其解码为完整的轨迹预测。这会生成一个形状为 `(B, K, T, 2)` 的张量，其中 K 为 `sample_k`（默认为 20）。[ADE 和 FDE 指标](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/metrics/ade_fde.py#L10-L26)随后计算每位行人在所有 K 个样本中的最小误差——选择最接近真实值的预测——再对批次取平均。

| 参数         | 默认值 | 作用                                 |
| ------------ | ------ | ------------------------------------ |
| `sample_k`   | 20     | 验证期间从前验分布中提取的轨迹样本数 |
| `beta`       | 1.0    | 训练期间 CVAE 损失中的 KL 散度权重   |
| `batch_size` | 32     | 训练和验证 DataLoader 的批次大小     |
| `lr`         | 0.001  | Adam 学习率                          |

验证返回两个标量：**val_ade** 和 **val_fde**。只有 `val_ade` 驱动检查点选择逻辑——这是一个刻意的设计选择，因为 ADE 捕捉了整个预测范围内的平均误差，使其成为比 FDE（仅考虑最终时间步）更稳定的整体模型质量指标。

来源: [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L50-L67), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L119-L126), [ade_fde.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/metrics/ade_fde.py#L10-L26)

## 检查点策略：最佳与最新

检查点系统实现了**双持久化**策略，保存两个不同的快照：`best.pt`（迄今为止验证 ADE 最低的模型）和 `last.pt`（最终周期结束时的模型）。这种分离对于损失景观可能存在噪声的 CVAE 模型至关重要——最佳检查点捕获了峰值性能，而最新检查点则支持训练恢复。



```
best_ade = float("inf")for epoch in range(cfg["train"]["epochs"]):    loss = train_one_epoch(model, train_loader, optimizer, device, cfg["train"]["beta"])    val_ade, val_fde = evaluate(model, val_loader, device, cfg["eval"]["sample_k"])     if val_ade < best_ade:        best_ade = val_ade        save_checkpoint(            {                "model": model.state_dict(),                "optimizer": optimizer.state_dict(),                "epoch": epoch,                "val_ade": val_ade,                "val_fde": val_fde,            },            save_dir / "best.pt",        )
```

每当 `val_ade` 有所改善时，`best.pt` 检查点就会被**原地**保存，这意味着它始终反映训练过程中遇到的最佳模型。检查点字典同时存储了模型和优化器的状态字典，从而实现完整的训练恢复。它还记录了周期数以及该周期的验证指标，这对训练后分析非常有用。

在周期循环结束后，`last.pt` 会被无条件保存：



```
last_path = save_dir / "last.pt"save_checkpoint(    {        "model": model.state_dict(),        "optimizer": optimizer.state_dict(),        "epoch": cfg["train"]["epochs"] - 1,    },    last_path,)
```

注意 `last.pt` 省略了 `val_ade` 和 `val_fde`——它主要用于训练恢复而非评估。接着，一个兜底保护机制会检查 `best.pt` 是否存在；如果不存在（可能由于 Bug 导致第一个周期的 ADE 已经是 `inf`），它会将最终模型保存为 `best.pt`，以确保下游评估脚本总能找到有效的检查点。

[utils/io.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/utils/io.py#L1-L25) 中的 I/O 实用工具刻意保持极简：

| 函数                                  | 用途                                             |
| ------------------------------------- | ------------------------------------------------ |
| `ensure_dir(path)`                    | 递归创建父目录；返回解析后的 `Path`              |
| `save_checkpoint(state, path)`        | 对父目录调用 `ensure_dir`，然后执行 `torch.save` |
| `load_checkpoint(path, map_location)` | 封装 `torch.load` 并带有可选的设备映射           |





恢复训练时，请从检查点同时加载 `model` 和 `optimizer` 的状态字典。优化器状态包含动量缓冲区（Adam 的一阶和二阶矩），这对于稳定收敛至关重要——使用全新的优化器恢复实际上重置了这些累加器，可能导致损失出现短暂的飙升。



来源: [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L131-L171), [io.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/utils/io.py#L9-L24)

## 可复现性：随机种子与确定性

在任何模型或数据构建之前，训练脚本会调用 `set_seed(cfg["train"]["seed"])` 以在整个技术栈中建立确定性行为。[utils/seed.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/utils/seed.py#L9-L16) 中的实现为四个独立的随机来源设置种子，并配置了两个 PyTorch 后端标志：



```
def set_seed(seed: int = 42) -> None:    random.seed(seed)    np.random.seed(seed)    torch.manual_seed(seed)    torch.cuda.manual_seed_all(seed)    torch.backends.cudnn.deterministic = True    torch.backends.cudnn.benchmark = False
```

前四行分别为 Python 的 `random`、NumPy、PyTorch CPU 和 PyTorch CUDA 随机数生成器设置种子。最后两行同样重要：`cudnn.deterministic = True` 强制 cuDNN 使用确定性的卷积算法（以微小的性能损失为代价），而 `cudnn.benchmark = False` 阻止 cuDNN 根据输入大小自动调整算法选择，否则在批次组合略有不同时，会在不同运行间引入不确定性。

这对于 CVAE 验证期间的潜在采样尤为重要——如果没有固定的种子，来自前验分布的 `sample_k` 次采样将在不同运行间产生差异，使得 ADE/FDE 的比较失去意义。

来源: [seed.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/utils/seed.py#L9-L16), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L76)

## 配置驱动的实验变体

训练脚本被设计为**可切换配置**：仅需更改 `--config` 参数即可替换整个实验设置。这是消融研究背后的机制，即通过四个 YAML 文件切换 `use_scene` 和 `use_social` 来控制哪些模型组件处于激活状态：

| 配置文件                 | `use_scene` | `use_social` | `save_dir`                     | 用途                 |
| ------------------------ | ----------- | ------------ | ------------------------------ | -------------------- |
| `ablation_baseline.yaml` | false       | false        | `outputs/checkpoints/baseline` | 仅 LSTM 编码器       |
| `ablation_social.yaml`   | false       | true         | `outputs/checkpoints/social`   | LSTM + Social GAT    |
| `ablation_scene.yaml`    | true        | false        | `outputs/checkpoints/scene`    | LSTM + Scene Encoder |
| `ablation_full.yaml`     | true        | true         | `outputs/checkpoints/full`     | 完整 CSGATNet        |

每个消融配置都会写入一个**独立的检查点目录**，防止意外覆盖，并支持跨实验直接对比 `best.pt` 文件。`default.yaml` 等同于 `ablation_full.yaml`，写入至 `outputs/checkpoints`。

来源: [ablation_baseline.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_baseline.yaml#L1-L45), [ablation_full.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_full.yaml#L1-L45), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L1-L45)

## 用于评估的检查点加载

配套的 [eval.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/eval.py#L66-L67) 脚本展示了检查点消费模式。它使用带有 `map_location` 参数的 `load_checkpoint` 加载检查点，以确保模型权重被放置在正确的设备上（在 GPU 上训练但在 CPU 上评估时尤为关键），然后仅恢复模型状态字典：



```
ckpt = load_checkpoint(cfg["eval"]["checkpoint"], map_location=device)model.load_state_dict(ckpt["model"])
```

保存（完整状态）与加载（仅模型）之间的这种分离是有意为之——评估绝不需要优化器状态，而训练恢复由单独的工作流处理。YAML 配置中的 `eval.checkpoint` 字段指向特定的 `best.pt` 路径，该路径因消融变体而异。

来源: [eval.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/eval.py#L66-L67), [eval.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/eval.py#L25-L53)

## 关键设计决策与权衡

训练循环做出了几项值得理解的刻意架构选择：

**原地最佳检查点** — 每当 `val_ade` 改善时，`best.pt` 文件就会被覆盖。这意味着你无法恢复后来被超越的先前最佳模型。对于 ADE 可能波动的长时间训练运行，可考虑添加周期性检查点保存（例如，每 N 个周期）作为扩展。

**无学习率调度** — 优化器是在整个训练过程中保持固定学习率的原生 Adam。对于 CVAE 模型，这通常已足够，因为 KL 退火（由 `beta` 控制）实际上起到了课程学习的作用。如果你观察到训练不稳定，学习率预热或余弦调度将是首选的干预措施。

**无梯度裁剪** — 反向传播直接应用梯度而不进行裁剪。对于带有自回归解码器的序列模型，梯度裁剪（例如，使用 max_norm=1.0 的 `torch.nn.utils.clip_grad_norm_`）可以防止梯度爆炸，特别是在训练早期周期。

**验证 ADE 作为唯一选择标准** — 仅由 ADE 驱动检查点选择。在端点精度比平均精度更重要的场景中（例如，碰撞避免），你可以将标准切换为 FDE 或两者的加权组合。

来源: [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L28-L47), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L129-L151)

## 后续内容

训练循环是 CVAE 损失与评估指标交汇的集成点。要了解各项背后的具体机制：

- **[CVAE 损失函数](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/13-cvae-loss-function)** — 深入探讨重建损失与 KL 散度如何结合，以及 `beta` 如何控制潜在空间结构。
- **[基于 Best-of-K 的 ADE/FDE 指标](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/14-ade-fde-metrics-with-best-of-k)** — Best-of-K 协议如何选择最接近的预测，以及为何它对多模态评估至关重要。
- **[配置与消融设置](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/15-configuration-and-ablation-setup)** — 四个 YAML 配置如何构建消融研究，以及每个变体隔离了什么。



# CVAE 损失函数

条件变分自编码器（CVAE）的损失函数是驱动模型通过学习结构化隐空间来生成**多模态轨迹预测**的训练目标。它平衡了两个相互竞争的目标：**重建保真度**（预测轨迹与真实值的匹配程度）与**隐空间正则性**（后验分布与先验分布的对齐程度）。这种由 β 加权的 KL 散度项所控制的权衡，使得模型能够在推理阶段采样出多样且合理的未来轨迹——这是确定性模型从根本上无法实现的能力。

## 损失架构概述

总的 CVAE 损失是两项的加权和：

LCVAE=Lrecon+β⋅DKL ⁣(qϕ(z∣c,f)  ∥  pψ(z∣c))LCVAE=Lrecon+β⋅DKL(qϕ(z∣c,f)∥pψ(z∣c))

其中 **c** 是条件向量（轨迹 + 场景 + 社交特征），**f** 是编码后的未来轨迹，**β** 是可配置的权重。下图展示了这两项如何与模型的内部组件相连：



来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L85-L127), [cvae_loss.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/losses/cvae_loss.py#L22-L33), [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L34-L42)

## 重建损失 — 均方误差

重建损失衡量的是预测未来轨迹与真实值之间的逐点差异。它被计算为所有时间步和坐标维度上的**纯均方误差（MSE）**：



```
def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:    return torch.mean((pred - target) ** 2)
```

这是最直观的组成部分：解码器生成形状为 `(B, T_pred, 2)` 的轨迹 `pred`，其中 `T_pred=12`，该损失计算每个预测坐标的平方位移平均值。与某些使用 L1 或 Huber 损失的轨迹预测工作不同，此实现使用了 **L2 损失**，对较大偏差的惩罚更重——这是一种隐式偏好，倾向于生成整体贴近真实值的预测，而不是允许偶尔出现较大的异常值。

一个重要的细节是：在训练期间，**重建损失使用的是从后验样本 z 解码出的单条轨迹**，而不是 K 选优策略。这意味着重建项直接优化了模型在获取从已知后验中采样的隐编码后，生成接近真实值轨迹的能力。K 选优评估策略仅用于验证指标（参见 [ADE/FDE 指标与 K 选优策略](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/14-ade-fde-metrics-with-best-of-k)）。

来源: [cvae_loss.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/losses/cvae_loss.py#L6-L7), [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L113-L117)

## KL 散度 — 后验与先验的相遇

KL 散度项强制要求已“窥探”过未来轨迹的学习后验分布 (q_\phi(z \mid c, f)) 保持在仅能看到条件特征的先验分布 (p_\psi(z \mid c)) 附近。这种对齐至关重要，因为在推理时无法获取未来信息，模型必须仅从先验中进行采样。

该实现计算了**两个对角高斯分布之间的解析 KL 散度**：



```
def kl_divergence(    mu_q: torch.Tensor,    logvar_q: torch.Tensor,    mu_p: torch.Tensor,    logvar_p: torch.Tensor,) -> torch.Tensor:    var_q = torch.exp(logvar_q)    var_p = torch.exp(logvar_p)    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)    return torch.mean(torch.sum(kl, dim=-1))
```

这遵循了具有对角协方差的多变量高斯分布之间 KL 散度的闭式表达式：

DKL(q∥p)=12∑d=1D[log⁡σp2σq2+σq2+(μq−μp)2σp2−1]DKL(q∥p)=21∑d=1D[logσq2σp2+σp2σq2+(μq−μp)2−1]

该实现有两个值得注意的关键细节。首先，**在隐维度上的求和**（`torch.sum(kl, dim=-1)`）计算了所有 `latent_dim=16` 个维度的联合 KL，将每个维度视为独立。其次，**在批次上的平均**（`torch.mean`）产生了一个标量损失。这种先求和再平均（而非先平均再求和）的顺序很重要：它确保了首先计算每个样本的总 KL 贡献，防止批次平均稀释单个样本的散度信号。

| 符号                | 形状              | 含义                               |
| ------------------- | ----------------- | ---------------------------------- |
| `mu_q` / `logvar_q` | `(B, latent_dim)` | 后验均值和对数方差（来自后验网络） |
| `mu_p` / `logvar_p` | `(B, latent_dim)` | 先验均值和对数方差（来自先验网络） |
| `kl` (中间值)       | `(B, latent_dim)` | 每个维度的 KL 值                   |
| 输出                | `()`              | 标量：批次上求和 KL 的平均值       |





`logvar` 参数化（存储 log σ² 而不是 σ²）在数值上更具优势：它保证了方差 `exp(logvar)` 始终为正而无需截断操作，并且通过 `exp` 的梯度流比在无约束的 σ 参数上进行 `square` 运算更稳定。



来源: [cvae_loss.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/losses/cvae_loss.py#L10-L19)

## β 权重 — 控制隐容量

`total_cvae_loss` 函数通过一个可配置的权重将两项结合：



```
def total_cvae_loss(    pred, target,    prior_mu, prior_logvar,    post_mu, post_logvar,    beta: float = 1.0,) -> torch.Tensor:    recon = reconstruction_loss(pred, target)    kl = kl_divergence(post_mu, post_logvar, prior_mu, prior_logvar)    return recon + beta * kl
```

**β 参数**直接控制隐空间的有效容量以及多样性与准确性的权衡：

| β 值            | 对训练的影响                       | 预测行为                                           |
| --------------- | ---------------------------------- | -------------------------------------------------- |
| β = 0           | 忽略 KL 项；后验无约束             | 后验退化为点估计；推理时先验无用；多模态预测效果差 |
| β = 1.0（默认） | 标准 VAE ELBO；平衡的权衡          | 先验匹配后验；预测结果多样且有理有据               |
| β > 1.0         | 更强的 KL 惩罚；存在“后验坍缩”风险 | 隐空间过度正则化；样本趋于相似；牺牲了多样性       |

在本项目中，**所有配置变体均使用 β = 1.0**（标准 ELBO），如训练配置 `cfg["train"]["beta"]` 中所指定。Beta 值从配置传递至训练循环再到损失函数，这使得进行 β-VAE 调度或循环退火策略的实验变得非常直接：



```
loss = total_cvae_loss(pred, fut, prior_mu, prior_logvar, post_mu, post_logvar, beta)
```





如果你观察到模型在推理时对 K 个样本生成了几乎相同的轨迹，这就是**后验坍缩**的迹象——KL 项占据了主导地位，隐编码 z 携带的信息极少。一种实用的补救措施是使用 KL 退火（在最初的几个 epoch 中将 β 从接近 0 线性增加到 1.0），在正则化介入之前，给重建项留出建立有用隐结构的时间。



来源: [cvae_loss.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/losses/cvae_loss.py#L22-L33), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L28-L47), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L29-L30)

## 损失如何与完整模型连接

孤立地理解损失函数是不够的——关键问题是五个损失输入（`pred`、`target`、`prior_mu/logvar`、`post_mu/logvar`）如何从模型的前向传播流至损失计算。下表追踪了每个变量的来源：

| 损失输入                    | `CSGATNet.forward()` 中的来源                                | 形状         | 生成者             |
| --------------------------- | ------------------------------------------------------------ | ------------ | ------------------ |
| `pred`                      | `self._decode(obs_traj[:, -1], cond_feat, z)`                | `(B, 12, 2)` | 自回归 LSTM 解码器 |
| `target`（= `fut`）         | 直接来自数据加载器                                           | `(B, 12, 2)` | 真实未来轨迹       |
| `prior_mu` / `prior_logvar` | `self.cvae(cond_feat, fut_feat)` → `self.prior(cond_feat)` 的切分 | 各 `(B, 16)` | 先验 MLP 网络      |
| `post_mu` / `post_logvar`   | `self.cvae(cond_feat, fut_feat)` → `self.posterior(cat([cond_feat, fut_feat]))` 的切分 | 各 `(B, 16)` | 后验 MLP 网络      |

**训练与推理之间的不对称性**是关键的架构洞见。在训练期间，提供了 `fut_traj`，因此 CVAE 头部会生成先验和后验分布，通过重参数化技巧从后验中采样 z，并且损失函数联合优化所有五个输出。在推理期间，`fut_traj=None`，后验被完全绕过，z 从先验中采样——KL 正则化确保了先验已学会生成有意义的隐编码。

来源: [csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L85-L127), [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L34-L42)

## 重参数化技巧与梯度流

CVAE 头部的 `sample` 方法实现了**重参数化技巧**，这对于使 CVAE 损失可微至关重要：



```
@staticmethoddef sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:    eps = torch.randn_like(mu)    return mu + eps * torch.exp(0.5 * logvar)
```

该方法没有直接从 ( \mathcal{N}(\mu, \sigma^2) ) 中采样 z——这会阻断梯度流，因为采样是一个随机的、不可微的操作——而是将计算重构为 ( z = \mu + \epsilon \cdot \sigma )，其中 ( \epsilon \sim \mathcal{N}(0, I) )。由于 `eps` 被视为外部噪声（无需对其求梯度），梯度可以顺畅地流过 `mu` 和 `logvar` 到达后验网络的参数。因子 `exp(0.5 * logvar)` 将对数方差转换为标准差：`0.5 * logvar = log(σ)`，因此 `exp(0.5 * logvar) = σ`。

这就是为什么 KL 散度可以解析地计算，而不是通过蒙特卡洛采样来估计——重参数化确保了重建损失（通过采样的 z → 解码器 → pred 路径）和 KL 损失（直接来自 mu/logvar）都拥有精确且低方差的梯度。

来源: [cvae_head.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/cvae_head.py#L29-L32)

## 端到端训练集成

损失计算作为简洁的单行调用嵌入在训练循环中。完整的每 epoch 训练周期如下进行：



具体的代码路径：



```
# 步骤 1: 前向传播 — 模型返回 pred + 四个分布参数pred, prior_mu, prior_logvar, post_mu, post_logvar = model(obs, fut, scene, neigh, mask) # 步骤 2: 计算组合损失loss = total_cvae_loss(pred, fut, prior_mu, prior_logvar, post_mu, post_logvar, beta) # 步骤 3: 标准 PyTorch 优化步骤optimizer.zero_grad()loss.backward()optimizer.step()
```

`beta` 值从 `cfg["train"]["beta"]` 中读取，并通过 `train_one_epoch` 的函数签名传递。本项目中的四种消融配置（基线、仅场景、仅社交、完整）均共享相同的 β = 1.0，确保消融结果反映的是特征编码的差异，而非损失权重的差异。

来源: [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L28-L47), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L134-L136)

## 对比：训练损失与评估指标

一个常见的混淆点是 CVAE 训练损失与评估指标（ADE/FDE）之间的关系。它们服务于根本不同的目的：

| 方面         | CVAE 训练损失        | ADE/FDE 评估指标       |
| ------------ | -------------------- | ---------------------- |
| **目的**     | 优化模型参数         | 衡量预测质量           |
| **重建**     | 单个后验样本上的 MSE | K 选优位移             |
| **KL 项**    | 包含（正则化）       | 不适用                 |
| **采样**     | z ~ 后验（已知信息） | z ~ 先验（对未来未知） |
| **轨迹数量** | 每个样本 1 个预测    | 每个样本 K=20 个预测   |
| **可微性**   | 是（驱动反向传播）   | 否（仅用于评估）       |

训练损失旨在优化**后验下的单样本重建质量**，而评估指标衡量的是**先验下的多样本预测质量**。KL 散度项是使先验变得有用的桥梁——如果没有它，就无法保证先验样本能生成有意义的轨迹。

有关如何使用 K 选优策略计算 ADE/FDE 的详细信息，请参见 [ADE/FDE 指标与 K 选优策略](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/14-ade-fde-metrics-with-best-of-k)。有关在消融实验中如何配置损失超参数的详细信息，请参见 [配置与消融设置](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/15-configuration-and-ablation-setup)。



# 基于 Best-of-K 的 ADE/FDE 指标

轨迹预测模型必须根据其预测路径与真实运动的贴合程度进行评估。本文剖析了本项目实现的两个事实标准指标——**ADE**（平均位移误差）和 **FDE**（最终位移误差），并解释了将随机 CVAE 采样与确定性基准比较联系起来的 **Best-of-K** 策略。理解这些指标对于解读消融实验结果、调整 `sample_k` 以及诊断轨迹时间范围内的预测质量至关重要。

来源：[ade_fde.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/metrics/ade_fde.py#L1-L27), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L37-L40)

## 指标定义与几何直觉

ADE 和 FDE 均源于**欧几里得位移**——即在给定时间步，预测坐标与对应真实坐标之间的 L2 距离。其核心原语是 `_pairwise_displacement`，它沿最后一个维度（2D 坐标平面）计算 `torch.norm(pred - target, dim=-1)`。这会生成一个逐时间步的位移张量，随后两个指标会以不同的方式对其进行降维。

**ADE** 对所有预测时间步的位移求平均值，衡量*整体轨迹保真度*。它反映了模型的预测路径是始终跟踪真实轨迹，还是逐渐发生漂移。**FDE** 仅提取最后时间步的位移，衡量*终点准确度*。它反映了模型是否正确预测了行人的最终到达位置——这对于避撞等下游任务至关重要，因为在这类任务中，目的地比中间路径更重要。

| 指标    | 降维方式                | 衡量内容                     | 敏感度                           |
| ------- | ----------------------- | ---------------------------- | -------------------------------- |
| **ADE** | 对所有 T 个预测步求均值 | 整个时间范围内的平均路径偏差 | 对系统性漂移敏感；抑制单步异常值 |
| **FDE** | 仅选择最后时间步        | 目的地预测准确度             | 对累积误差敏感；反映长期预测质量 |

对于形状为 `(B, T, 2)` 的单轨迹预测，ADE 计算 `disp.mean()`，FDE 计算 `disp[:, -1].mean()`，两者最终均在批次维度上求平均作为最终降维。

来源：[ade_fde.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/metrics/ade_fde.py#L6-L27)

## Best-of-K 评估策略

CVAE 架构会产生**多模态预测**：在推理阶段，先验网络对潜在向量 `z ~ N(μ_prior, σ²_prior)` 进行采样，每个样本通过自回归解码器生成一条独立的轨迹。单次采样可能会遗漏真实的模态，因此标准评估协议会抽取 **K 个独立样本**，并为每位行人选择最接近真实轨迹的样本。这便是 **Best-of-K** 策略。

该机制与模型的推理路径紧密耦合。当 `fut_traj is None`（评估模式）时，`CSGATNet.forward` 会循环执行 `sample_k` 次，每次从先验分布中抽取一个新的 `z` 并解码出完整轨迹，随后将结果堆叠成形状 `(B, K, T, 2)`。指标模块检测到此 4D 张量后，会自动应用 Best-of-K 选择：



其关键的实现洞察在于**广播技巧**：对于 4D 预测张量，形状为 `(B, T, 2)` 的真实目标 `target` 会通过 `target[:, None, :, :]` 扩展为 `(B, 1, T, 2)`，从而能够同时与所有 K 个候选轨迹进行逐元素位移计算。在计算出每个样本、每个时间步的位移后，ADE 先在时间维度上执行 `mean(dim=-1)` 求均值，再在样本维度上执行 `min(dim=1)` 取最小值；而 FDE 先选取最后时间步，再在样本维度上应用 `min(dim=1)`。

来源：[csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L119-L126), [ade_fde.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/metrics/ade_fde.py#L10-L26)

## 张量形状约定与分派逻辑

`ade()` 和 `fde()` 函数使用**张量维度作为分派信号**——这是一种刻意的设计，将单轨迹和 Best-of-K 评估统一在同一 API 下，避免了单独的函数调用或包装类。

| 输入形状       | 解释           | ADE 路径                                         | FDE 路径                                      |
| -------------- | -------------- | ------------------------------------------------ | --------------------------------------------- |
| `(B, T, 2)`    | 单条确定性轨迹 | `disp.mean()`                                    | `disp[:, -1].mean()`                          |
| `(B, K, T, 2)` | K 个随机样本   | `disp.mean(dim=-1)` → `min(dim=1).values.mean()` | `disp[:, :, -1]` → `min(dim=1).values.mean()` |





当 `pred.dim() == 4` 时，FDE 实现会在计算位移之前通过 `pred[:, :, -1]` 提取*最后时间步*，这提前折叠了 T 维度，避免了在中间时间步上进行不必要的计算。这既是一种内存优化，也是逻辑清晰的体现：FDE 只关心终点。







`min(dim=1).values` 操作通过为每个批次元素独立选择*位移最小的样本*来实现 Best-of-K。这意味着不同的行人可能在不同的 K 索引上拥有各自的“最佳”样本——每条轨迹都是与其最接近的预测进行评估，而非共享全局统一的最佳预测。



来源：[ade_fde.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/metrics/ade_fde.py#L10-L26)

## 与训练和评估流程的集成

根据流程阶段的不同，这些指标发挥着两种不同的作用。在**训练**期间，`train_one_epoch` 使用 CVAE 重建损失（MSE）而非 ADE/FDE 进行梯度更新——由于 `min` 操作和自回归累积的存在，ADE/FDE 是不可微的。训练循环仅在每个 epoch 结束后调用 `evaluate()`，该函数在 `sample_k` 模式下运行模型，并将 ADE/FDE 作为检查点选择的*监控信号*进行上报。

在**独立评估**期间，`scripts/eval.py` 遵循相同的模式：加载最佳检查点，使用 `model(obs, None, scene, neigh, mask, sample_k=cfg["eval"]["sample_k"])` 迭代验证集，累积每个批次的 ADE/FDE，并报告数据集级别的均值。只要 `val_ade < best_ade` 成立，就会保存检查点，这使得 ADE 成为了主要的模型选择标准。



`sample_k` 参数在配置中的默认值为 **20**，意味着在评估期间每位行人生成 20 个潜在样本。该值在指标可靠性与计算成本之间取得了平衡：样本过少会导致对先验分布采样不足，遗漏相关模态；而在主导模态已被捕获后，样本过多只会增加延迟，而不会带来相应比例的指标提升。

来源：[train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L50-L67), [eval.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/eval.py#L70-L86), [default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L38-L40)

## 透过指标视角解读消融结果

消融结果揭示了模型组件与指标行为之间微妙的相互作用：

| 模型                | ADE    | FDE    | 解释                                         |
| ------------------- | ------ | ------ | -------------------------------------------- |
| 基线（仅 LSTM）     | 0.0132 | 0.0247 | 无社会或场景引导；存在适度漂移               |
| 仅场景              | 0.0196 | 0.0325 | 场景上下文在缺乏社会基础的情况下仅增加了噪声 |
| 仅社会              | 0.0118 | 0.0210 | 最佳 ADE——社会交互减少了平均路径偏差         |
| 完整（场景 + 社会） | 0.0137 | 0.0184 | 最佳 FDE——场景语义锚定了最终目的地           |

ADE 最优（仅社会）与 FDE 最优（完整模型）之间的分歧具有架构层面的指导意义。**仅社会**模型在 ADE 上表现优异，因为对邻居轨迹的注意力平滑了每一步的预测路径，减少了逐时间步的漂移。**完整**模型在 FDE 上胜出，因为场景编码器提供了空间先验（如可通行区域、障碍物），即使中间的社会注意力引入了轻微扰动，这些先验也能约束*终点*。这种模式——即添加场景特征会略微增加 ADE，但会降低 FDE——是场景语义充当**目的地正则化器**而非路径平滑器的模型的典型特征。

来源：[ablation_results.md](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/references/ablation_results.md#L1-L14)

## 指标使用的实际考量

在你自己的实验中使用这些指标时，有几个实现细节值得关注。首先，这些指标基于**坐标空间位移**进行计算，而非归一化或缩放后的值——请确保预测值与真实值共享同一坐标系（[ETH/UCY 数据集预处理](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/9-eth-ucy-dataset-preprocessing)中的预处理流程负责处理此归一化）。其次，Best-of-K 的 `min` 操作引入了**乐观偏差**：随着 K 的增加，指标会单调改善，因为更多的样本增加了其中一个落在真实轨迹附近的概率。在比较模型时，务必使用相同的 `sample_k` 值；将 K=5 的模型与 K=20 的模型进行比较会产生误导。第三，当前的实现是先*按批次*计算指标平均值，然后再对所有批次求平均（`sum(ade_list) / len(ade_list)`），只有当所有批次大小相同时，这才等同于均匀加权平均——鉴于 DataLoader 具有一致的 `batch_size`，这是一个合理的假设，但如果最后一个批次较小且你转而采用加权平均方案，该假设便会失效。

来源：[eval.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/eval.py#L84-L86), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L67)

## 后续步骤

- 了解产生这些指标所评估的预测结果的训练目标：[CVAE 损失函数](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/13-cvae-loss-function)
- 查看 `sample_k` 及其他评估参数的配置方式：[配置与消融设置](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/15-configuration-and-ablation-setup)
- 探索多模态预测的可视化输出：[预测可视化流程](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/16-prediction-visualization-pipeline)



# 配置与消融设置

本页解释了项目基于 YAML 的配置系统的工作原理，如何通过简单的布尔开关组织消融实验，以及每个配置变体如何从文件流向模型构建。理解这些设置对于复现实验、扩展模型或运行你自己的消融研究至关重要。

## 配置架构

所有实验设置均存放在 `configs/` 目录下的 YAML 文件中。训练脚本 [`scripts/train.py`](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py) 接受一个指向这些文件之一的 `--config` 参数，然后将每个值传递给数据集构造器、模型构造器、优化器和评估循环。这种设计将所有实验变量集中在一处，使得复现或对比运行变得轻而易举。

配置由五个顶层部分组成，每个部分控制流水线的一个不同阶段：

| 部分        | 用途                           | 关键参数                                                     |
| ----------- | ------------------------------ | ------------------------------------------------------------ |
| `data`      | 数据集路径、轨迹长度、邻居构建 | `train_dir`, `val_dir`, `obs_len`, `pred_len`, `neighbor_radius`, `max_neighbors` |
| `model`     | 网络维度及**模块开关**         | `hidden_dim`, `scene_dim`, `latent_dim`, `use_scene`, `use_social` |
| `train`     | 优化超参数和检查点路径         | `batch_size`, `epochs`, `lr`, `beta`, `seed`, `save_dir`     |
| `eval`      | 推理时的采样                   | `sample_k`, `checkpoint`                                     |
| `visualize` | 生成图表的输出目录             | `output_dir`                                                 |





`model.use_scene` 和 `model.use_social` 这两个布尔标志是四个消融配置之间的唯一区别。其他所有内容——维度、学习率、数据路径——均保持一致，从而确保了实验的受控性。



来源：[default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L1-L45), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L56-L100)

### 默认配置

[`configs/default.yaml`](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml) 文件表示启用了场景和社交模块的**完整模型**。以下是各参数控制内容的详细说明：

**Data 部分** — `obs_len: 8` 和 `pred_len: 12` 定义了标准的 ETH/UCY 数据集划分（观测 8 帧，预测 12 帧）。`neighbor_radius: 2.0`（单位：米）决定了哪些附近行人符合社交邻居的条件，而 `max_neighbors: 16` 限制了邻居数量以控制内存占用。`scene_dir: null` 标志表示默认不加载预提取的语义图；数据集类会妥善处理此情况，为场景输入返回 `None`。

**Model 部分** — `hidden_dim: 64` 是轨迹编码器、邻居编码器和解码器共享的 LSTM 隐藏层大小。`scene_dim: 128` 是 ResNet 场景编码器的输出维度。`latent_dim: 16` 控制 CVAE 潜在向量的大小。两个布尔标志 `use_scene: true` 和 `use_social: true` 用于激活各自的模块。

**Train 部分** — `beta: 1.0` 是 CVAE 损失中 KL 散度的权重（参见 [CVAE 损失函数](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/13-cvae-loss-function)）。`seed: 42` 在训练开始时通过 [`utils/seed.py`](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/utils/seed.py) 设置，以固定 Python、NumPy 和 PyTorch 的随机状态，从而保证可复现性。

来源：[default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L1-L45), [seed.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/utils/seed.py#L1-L17)

### 配置如何流入代码

下图展示了单个 YAML 文件如何驱动整个训练流水线：



关键的设计模式在于 [`CSGATNet.__init__`](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L26-L52) 会根据这两个布尔标志**按条件实例化**子模块。如果 `use_scene` 为 `False`，则永远不会创建 `SceneEncoder`，并且 `fuse_dim` 会缩减 `scene_dim`。如果 `use_social` 为 `False`，则省略 `SocialGAT`，并且 `fuse_dim` 会再缩减一个 `hidden_dim`。这意味着参数量会自动调整以匹配对应的消融变体——模型中不存在死权重或未使用的计算路径。

来源：[csgat_net.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/models/csgat_net.py#L26-L52), [train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L82-L106)

## 消融实验设计

消融研究通过系统性地启用或禁用每个感知模块——**场景理解**和**社交交互**——来隔离它们的贡献。共定义了四种配置：

| 配置文件                 | `use_scene` | `use_social` | 激活模块              | 描述                       |
| ------------------------ | ----------- | ------------ | --------------------- | -------------------------- |
| `ablation_baseline.yaml` | false       | false        | 仅 LSTM 编码器        | 纯轨迹外推——无上下文       |
| `ablation_scene.yaml`    | true        | false        | LSTM + SceneEncoder   | 无社交推理的场景感知预测   |
| `ablation_social.yaml`   | false       | true         | LSTM + SocialGAT      | 无场景上下文的社交感知预测 |
| `ablation_full.yaml`     | true        | true         | LSTM + Scene + Social | 包含两个模块的完整模型     |

这些配置之间的关系构成了一个析因设计——两个二元因子（场景、社交）产生四种组合，使你能够测量每个模块的**边际贡献**及它们之间的任何**交互效应**。





每个消融配置都会将检查点写入 `outputs/checkpoints/` 下各自的子目录中（例如，`baseline/`、`scene/`、`social/`、`full/`）。这可以防止意外覆盖，并便于后续比较训练好的模型。



来源：[ablation_baseline.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_baseline.yaml#L1-L45), [ablation_scene.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_scene.yaml#L1-L45), [ablation_social.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_social.yaml#L1-L45), [ablation_full.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/ablation_full.yaml#L1-L45)

### 运行消融实验

使用指向相应 YAML 文件的单条命令即可启动每个消融变体：



```
python scripts/train.py --config configs/ablation_baseline.yaml # 仅场景 — 增加场景语义特征python scripts/train.py --config configs/ablation_scene.yaml # 仅社交 — 增加图注意力社交特征python scripts/train.py --config configs/ablation_social.yaml # 完整模型 — 同时包含场景和社交python scripts/train.py --config configs/ablation_full.yaml
```

进行评估时，请使用相同的配置文件，以便模型架构与检查点相匹配：



```
python scripts/eval.py --config configs/ablation_social.yaml
```

评估脚本会读取 `cfg.eval.checkpoint` 来查找正确的 `best.pt` 文件，并使用相同的 `use_scene` / `use_social` 标志重建模型，从而确保参数字典完美对齐。

来源：[train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L103-L118), [eval.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/eval.py#L48-L66)

### 消融结果

下表总结了所有四个变体的实验结果（两项指标均为越低越好）：

| 模型                      | `use_scene` | `use_social` | ADE ↓  | FDE ↓      |
| ------------------------- | ----------- | ------------ | ------ | ---------- |
| **基线** (仅 LSTM)        | ✗           | ✗            | 0.0132 | 0.0247     |
| **仅场景**                | ✓           | ✗            | 0.0196 | 0.0325     |
| **仅社交**                | ✗           | ✓            | 0.0118 | 0.0210     |
| **完整** (Scene + Social) | ✓           | ✓            | 0.0137 | **0.0184** |

从消融的角度解读这些结果：

- **社交交互提供了最强的单模块提升。** 仅社交变体取得了最佳的 ADE（0.0118）和优异的 FDE（0.0210），两者均优于基线。这证实了建模行人对行人的影响对轨迹准确性至关重要。
- **仅场景特征轻微损害了性能。** 仅场景变体（ADE 0.0196，FDE 0.0325）的表现逊于基线。这表明，在没有社交推理的情况下注入场景上下文时，模型可能会被那些不能直接约束短期行人运动的空间特征所误导。
- **完整模型取得了最佳的 FDE。** 结合两个模块产生了最低的最终位移误差（0.0184），这表明场景和社交特征提供了**互补信息**——特别是对于终点精度而言，物理约束（可行走区域、障碍物）在较长的时间跨度上变得更加相关。

来源：[ablation_results.md](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/references/ablation_results.md#L1-L14)

## 配置参数参考

为了便于快速查阅，以下是包含描述和默认值的完整参数目录：

### 数据参数

| 参数                   | 默认值                 | 类型      | 描述                               |
| ---------------------- | ---------------------- | --------- | ---------------------------------- |
| `data.train_dir`       | `data/processed/train` | str       | 已处理的训练轨迹 `.txt` 文件目录   |
| `data.val_dir`         | `data/processed/val`   | str       | 已处理的验证轨迹 `.txt` 文件目录   |
| `data.obs_len`         | 8                      | int       | 观测帧数（输入序列长度）           |
| `data.pred_len`        | 12                     | int       | 要预测的未来帧数                   |
| `data.skip`            | 1                      | int       | 构建样本时的滑动窗口步长           |
| `data.min_ped`         | 1                      | int       | 包含一个样本所需的每场景最少行人数 |
| `data.neighbor_radius` | 2.0                    | float     | 邻居纳入的最大距离（米）           |
| `data.max_neighbors`   | 16                     | int       | 每个行人的最大邻居数               |
| `data.scene_dir`       | null                   | str\|null | 预提取的场景语义图（`.npy`）路径   |
| `data.scene_ext`       | `.npy`                 | str       | 场景图文件的扩展名                 |

### 模型参数

| 参数               | 默认值 | 类型 | 描述                          |
| ------------------ | ------ | ---- | ----------------------------- |
| `model.hidden_dim` | 64     | int  | LSTM 隐藏层大小和轨迹特征维度 |
| `model.scene_dim`  | 128    | int  | ResNet 场景编码器的输出维度   |
| `model.latent_dim` | 16     | int  | CVAE 潜在向量维度             |
| `model.use_scene`  | true   | bool | 启用/禁用场景语义编码器       |
| `model.use_social` | true   | bool | 启用/禁用社交图注意力模块     |

### 训练参数

| 参数                 | 默认值                | 类型  | 描述                          |
| -------------------- | --------------------- | ----- | ----------------------------- |
| `train.batch_size`   | 32                    | int   | 小批量大小                    |
| `train.epochs`       | 20                    | int   | 总训练轮数                    |
| `train.lr`           | 0.001                 | float | Adam 学习率                   |
| `train.beta`         | 1.0                   | float | CVAE 损失中的 KL 散度权重     |
| `train.num_workers`  | 2                     | int   | DataLoader 工作线程数         |
| `train.seed`         | 42                    | int   | 用于可复现性的随机种子        |
| `train.device`       | `cuda`                | str   | 目标设备（`cuda` 或 `cpu`）   |
| `train.log_interval` | 20                    | int   | 每隔 N 个批次打印训练统计信息 |
| `train.save_dir`     | `outputs/checkpoints` | str   | 保存模型检查点的目录          |

### 评估参数

| 参数              | 默认值                        | 类型 | 描述                              |
| ----------------- | ----------------------------- | ---- | --------------------------------- |
| `eval.sample_k`   | 20                            | int  | 用于 Best-of-K 评估的 CVAE 采样数 |
| `eval.checkpoint` | `outputs/checkpoints/best.pt` | str  | 要评估的模型检查点路径            |

来源：[default.yaml](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/configs/default.yaml#L1-L45)

## 扩展配置

如果你想添加一个新的消融维度——例如改变 `latent_dim` 或 `beta`——工作流程非常简单：

1. **复制现有配置**作为模板（例如，`cp configs/ablation_full.yaml configs/ablation_latent32.yaml`）
2. **修改相关参数**（例如，将 `model.latent_dim` 改为 `32`）
3. **更新 `train.save_dir`** 以指向新的子目录（例如，`outputs/checkpoints/latent32`）
4. **更新 `eval.checkpoint`** 以匹配新的保存路径
5. **运行训练**：`python scripts/train.py --config configs/ablation_latent32.yaml`

需要维护的关键原则是**每个配置必须是自包含的**——训练和评估脚本从单个 YAML 文件中读取所有参数，没有隐藏的默认值。这确保了任何实验都可以仅凭其配置文件完全复现。

来源：[train.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/train.py#L56-L66), [eval.py](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/scripts/eval.py#L34-L44)

## 后续步骤

- 要了解 CVAE 损失如何使用 `beta` 参数，请参阅 [CVAE 损失函数](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/13-cvae-loss-function)
- 要查看 `sample_k` 如何输入 Best-of-K 评估协议，请参阅 [Best-of-K 的 ADE/FDE 指标](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/14-ade-fde-metrics-with-best-of-k)
- 要可视化不同消融变体的预测结果，请参阅 [预测可视化流水线](https://zread.ai/Flag64Zhang/Pedestrian-Trajectory-Prediction-with-Semantic-Segmentation---Graph-Attention/16-prediction-visualization-pipeline)




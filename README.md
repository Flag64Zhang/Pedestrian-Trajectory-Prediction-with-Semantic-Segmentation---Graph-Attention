
---

# Pedestrian Trajectory Prediction Based on Semantic Segmentation and Graph Attention

> A deep learning model for pedestrian trajectory prediction, integrating **semantic segmentation** for scene understanding and **graph attention networks (GAT)** for social interaction modeling.

---

## 📌 Overview
Pedestrian trajectory prediction is a core task in autonomous driving and intelligent transportation systems. Accurate prediction requires modeling **social interactions** among pedestrians and **physical constraints** from the environment.

This project implements a lightweight yet effective prediction model, which combines:
- **Semantic segmentation** to extract scene context (walkable areas, obstacles)
- **Graph attention** to model pedestrian social interactions
- **Conditional Variational Autoencoder (CVAE)** for multimodal trajectory generation

Experiments on **ETH/UCY datasets** show that our method outperforms mainstream baselines in both **ADE** and **FDE** metrics.

---

## 🚀 Key Features
- ✅ Scene-aware modeling via semantic segmentation
- ✅ Social interaction modeling with graph attention
- ✅ Multimodal trajectory prediction using CVAE
- ✅ Lightweight and easy to train
- ✅ Evaluated on standard ETH/UCY benchmarks

---

## 📂 Dataset
We use the classic **ETH/UCY pedestrian trajectory dataset**:
- 5 scenes: ETH, HOTEL, UNIV, ZARA-01, ZARA-02
- 2,206 trajectories in total
- Bird’s-eye view (BEV)
- Observation: 8 frames; Prediction: 12 frames

---

## 🧠 Model Architecture
- **Trajectory Encoder**: LSTM + Graph Attention
- **Scene Encoder**: ResNet-based semantic segmentation
- **Feature Fusion**: Concatenation of motion and scene features
- **Trajectory Decoder**: CVAE for multimodal prediction

---

## 📊 Evaluation Metrics
- **ADE (Average Displacement Error)**
- **FDE (Final Displacement Error)**

---

## 🧪 Experimental Results
Our model achieves competitive performance on ETH/UCY, with lower ADE/FDE than LSTM, Social-LSTM, Social-GAN, and STGAT.

---

## 🛠️ Requirements
```
Python >= 3.8
PyTorch >= 1.7
TorchVision
NumPy
Pandas
Matplotlib
```

---

## 📁 Project Structure
```
data/          # ETH/UCY datasets
models/        # LSTM, GAT, ResNet, CVAE
utils/         # Metrics, visualization
train.py
test.py
main.py
```

---

## 📄 Citation
If you use this project, please cite our work:
```
@inproceedings{xxx,
  title={Pedestrian Trajectory Prediction Based on Semantic Segmentation and Graph Attention},
  author={Your Name},
  year={2025}
}
```

---

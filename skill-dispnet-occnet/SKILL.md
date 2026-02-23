---
name: dispnet-occnet-reproduction
description: 复现论文《Unsupervised Light Field Depth Estimation via Multi-view Feature Matching with Occlusion Prediction》的完整技能。包含 DispNet（视差估计网络）和 OccNet（遮挡预测网络）的实现，用于无监督光场深度估计。使用多视图特征匹配、遮挡处理和视差融合策略。
---

# DispNet-OccNet 论文复现技能

本技能用于复现 IEEE TIP 论文 "Unsupervised Light Field Depth Estimation via Multi-view Feature Matching with Occlusion Prediction"（作者：Shansi Zhang, Nan Meng, Edmund Y. Lam）。

## 快速开始

### 1. 环境配置

```bash
# 使用配置脚本设置环境
python scripts/setup_environment.py
```

### 2. 准备数据集

支持的数据集：
- **HCI Dataset**（密集光场）：http://lightfield-analysis.net/dataset/
- **DLF Dataset**（密集光场）
- **SLF Dataset**（稀疏光场）
- **Stanford Lytro Archive**（真实世界光场）

### 3. 训练模型

```bash
# 密集光场训练
python train.py --config configs/dense_lf.yaml

# 稀疏光场训练
python train.py --config configs/sparse_lf.yaml
```

### 4. 推理和评估

```bash
# 单张图像推理
python inference.py --model_path checkpoints/best_model.pth --input_path input.png

# 批量评估
python evaluate.py --data_dir ./data/hci --split test
```

## 核心组件

### DispNet（视差估计网络）

DispNet 采用 coarse-to-fine 结构，通过多视图特征匹配学习视差。

**关键特性**：
- 三视图输入（中心视图 + 左右源视图）
- 基于方差的特征匹配构建代价体
- 共享权重的 cost filters
- 粗视差分支 + 残差 refinement 分支

查看详细信息：[references/disparity_estimation.md](references/disparity_estimation.md)

### OccNet（遮挡预测网络）

OccNet 预测遮挡图，用作 photometric loss 的像素级权重。

**关键特性**：
- U-Net 结构
- 输入：warp 后的左右视图 + 视差图
- 输出：左右视图的 confidence maps
- 仅在训练时使用

查看详细信息：[references/occlusion_prediction.md](references/occlusion_prediction.md)

### 损失函数

无监督训练使用以下损失：

1. **Weighted Photometric Loss** (ℓ_wpm)：使用遮挡图加权的 photometric 一致性
2. **Reconstruction Loss** (ℓ_rec)：OccNet 的重建损失
3. **SSIM Loss** (ℓ_SSIM)：结构相似性损失
4. **Smoothness Loss** (ℓ_smd)：视差平滑性损失（edge-aware）
5. **Occlusion Smoothness Loss** (ℓ_smo)：遮挡图平滑性损失

查看详细信息：[references/loss_functions.md](references/loss_functions.md)

### 视差融合策略

多视图组合估计多个视差图，基于估计误差进行融合：

1. 使用辅助视图计算 warping 误差
2. 检测遮挡区域（标准差阈值）
3. 遮挡区域使用中值，其他区域使用均值
4. Weighted fusion（选择误差最小的 2 个）

查看详细信息：[references/disparity_fusion.md](references/disparity_fusion.md)

## 复现步骤

### 第一步：理解论文方法

阅读 [references/paper_summary.md](references/paper_summary.md) 了解：
- 问题定义和光场几何
- 整体框架设计
- 核心创新点

### 第二步：实现 DispNet

按照以下顺序实现：
1. 特征提取器（Residual blocks + ASPP）
2. 方差基特征匹配构建 cost volume
3. Cost filters（3D residual blocks）
4. Disparity regression（soft argmin）
5. Coarse-to-fine refinement

参考：[references/network_architecture.md](references/network_architecture.md)

### 第三步：实现 OccNet

1. U-Net 编码器 - 解码器结构
2. 输入拼接（I_l→c, I_r→c, d̃）
3. Softmax 输出 confidence maps
4. 联合训练策略

参考：[references/occlusion_prediction.md](references/occlusion_prediction.md)

### 第四步：实现损失函数

实现完整的 loss 组合：
```python
ℓ_full = ℓ_wpm + ℓ_rec + α₁ℓ_SSIM + α₂ℓ_smd + α₃ℓ_smo
```

权重设置：
- α₁ = 1（SSIM）
- α₂ = 0.1（视差平滑）
- α₃ = 0.05（遮挡平滑）

参考：[references/loss_functions.md](references/loss_functions.md)

### 第五步：训练和验证

**密集光场配置**：
- 视差范围：[-12, 12]，间隔 1
- 残差范围：[-1, 1]，间隔 0.1
- 输入组合：6 种（距离 2 和 3 的视图）
- Batch size：4
- Learning rate：1e-3，每 50 epoch × 0.8
- Epochs：500

**稀疏光场配置**：
- 视差范围：[-20, 20]，间隔 1.2
- 残差范围：[-2, 2]，间隔 0.12
- 输入组合：2 种（相邻视图）
- 其他参数相同

参考：[references/training_guide.md](references/training_guide.md)

### 第六步：推理和融合

推理流程：
1. 多视图组合输入 DispNet
2. 获取多个视差估计（缩放 + 旋转）
3. 使用辅助视图计算误差图
4. 遮挡处理（中值/均值选择）
5. Weighted fusion（n'=2）

参考：[references/inference.md](references/inference.md)

## 关键参数

### 网络参数
- DispNet 参数量：1.802M
- OccNet 参数量：0.113M
- Feature extractor 最大 channel：128
- Cost filters 数量：2

### 超参数
- η = 100（smoothness loss）
- 遮挡检测 quantile q = 0.95
- Fusion 选择 n' = 2

## 预期结果

### HCI Dataset（MSE × 100 / BPR@0.07）
- Dino: 2.266 / 9.238
- Sideboard: 根据实验结果
- 优于其他无监督方法，接近监督方法

### 运行时间
- 密集光场：~2.688s（NVIDIA Tesla P100）
- 稀疏光场：~1.5s

## 常见问题

查看 [references/faq.md](references/faq.md) 解决：
- CUDA OOM 问题
- 损失不收敛
- 深度图质量差
- 评估指标计算

## 脚本工具

- `scripts/analyze_paper.py`：论文分析
- `scripts/setup_environment.py`：环境配置
- `scripts/download_dataset.py`：数据集下载
- `scripts/convert_dataset.py`：数据格式转换

## 配置文件

- `configs/dense_lf.yaml`：密集光场配置
- `configs/sparse_lf.yaml`：稀疏光场配置
- `configs/default.yaml`：默认配置

## 参考资源

### 论文相关
- [完整论文分析](references/paper_summary.md)
- [公式推导](references/formulas.md)
- [实验设置](references/experiments.md)

### 实现细节
- [网络架构](references/network_architecture.md)
- [数据处理](references/data_processing.md)
- [训练技巧](references/training_tips.md)

### 外部资源
- HCI Dataset: https://lightfield-analysis.net/
- Stanford Lytro: http://lightfields.stanford.edu/
- Papers With Code: 搜索相关方法

## 下一步

1. 阅读 [references/paper_summary.md](references/paper_summary.md) 理解论文
2. 运行 `scripts/setup_environment.py` 配置环境
3. 下载数据集并准备数据
4. 开始训练并监控进度
5. 评估结果并与论文对比

# DispNet-OccNet PyTorch

PyTorch 复现论文《Unsupervised Light Field Depth Estimation via Multi-view Feature Matching with Occlusion Prediction》

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIP%20202X-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## 📖 论文信息

**标题**: Unsupervised Light Field Depth Estimation via Multi-view Feature Matching with Occlusion Prediction

**作者**: Shansi Zhang, Nan Meng, Edmund Y. Lam

**发表**: IEEE Transactions on Image Processing

## 🎯 项目简介

本项目是论文《Unsupervised Light Field Depth Estimation via Multi-view Feature Matching with Occlusion Prediction》的 PyTorch 复现版本。

该论文提出了一种无监督光场深度估计方法，主要贡献包括：

1. **DispNet**: 采用 coarse-to-fine 结构的视差估计网络
2. **OccNet**: 遮挡预测网络，用于处理遮挡区域
3. **多视图特征匹配**: 基于方差的特征匹配构建代价体
4. **视差融合策略**: 基于误差的多视差融合

## ✅ 已完成

- [x] DispNet 网络实现
- [x] OccNet 网络实现
- [x] 损失函数实现
- [x] 数据加载器（支持 HCI 数据集）
- [x] 训练脚本
- [x] 配置文件
- [x] 代码编译测试通过
- [ ] 推理和评估

## 📦 安装

### 环境要求

- Python >= 3.7
- PyTorch >= 1.9.0
- torchvision >= 0.10.0

### 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 测试网络

```bash
python test_simple.py
```

### 训练模型（待实现）

```bash
python train.py --config configs/dense_lf.yaml
```

### 推理（待实现）

```bash
python inference.py --model_path checkpoints/best_model.pth --input_path input.png
```

## 📁 项目结构

```
dispnet-occnet-pytorch/
├── src/                    # 源代码
│   ├── __init__.py
│   ├── dispnet.py         # DispNet 网络
│   ├── occnet.py          # OccNet 网络
│   ├── loss.py            # 损失函数（待实现）
│   └── data.py            # 数据加载器（待实现）
├── configs/               # 配置文件
├── scripts/               # 工具脚本
├── data/                  # 数据集目录
├── checkpoints/           # 模型检查点
├── outputs/               # 输出目录
├── train.py               # 训练脚本（待实现）
├── test_simple.py         # 测试脚本
├── requirements.txt       # 依赖
└── README.md
```

## 📊 网络架构

### DispNet

```
输入：三视图 [I_left, I_center, I_right]
  ↓
特征提取器 (Residual blocks + ASPP)
  ↓
方差基特征匹配 → Cost Volume
  ↓
Coarse Cost Filters (3D residual blocks)
  ↓
Coarse Disparity Regression (soft argmin)
  ↓
Residual Cost Volume (使用 coarse disparity warp)
  ↓
Residual Cost Filters
  ↓
Residual Disparity Regression
  ↓
输出：d̃ = d̃_coarse + d̃_residual
```

**参数量**: ~1.8M

### OccNet

```
输入：[I_l→c, I_r→c, d̃] (7 channels)
  ↓
U-Net Encoder-Decoder
  ↓
Softmax
  ↓
输出：[O_l, O_r] (confidence maps)
```

**参数量**: ~0.11M

## 📈 预期结果

### HCI Dataset

| 场景 | MSE (×100) | BPR@0.07 |
|------|-----------|----------|
| Dino | 2.266 | 9.238 |

## 🔧 配置

### 密集光场配置

- 视差范围：[-12, 12]，间隔 1
- 残差范围：[-1, 1]，间隔 0.1
- 输入组合：6 种（距离 2 和 3 的视图）
- Batch size: 4
- Learning rate: 1e-3

### 稀疏光场配置

- 视差范围：[-20, 20]，间隔 1.2
- 残差范围：[-2, 2]，间隔 0.12
- 输入组合：2 种（相邻视图）

## 📝 待办事项

- [ ] 实现损失函数模块
- [ ] 实现数据加载器
- [ ] 创建训练脚本
- [ ] 创建配置文件
- [ ] 添加推理功能
- [ ] 添加评估指标
- [ ] 在 HCI 数据集上测试

## 🙏 致谢

- 论文作者和原始代码（如有）
- PyTorch 团队
- 光场研究社区

## 📄 许可证

本项目采用 MIT 许可证。

## 🔗 相关链接

- [论文 PDF](./Unsupervised%20Light%20Field%20Depth%20Estimation%20via.pdf)
- [HCI Dataset](http://lightfield-analysis.net/)
- [Stanford Lytro](http://lightfields.stanford.edu/)

## 📧 联系

如有问题，请提 Issue 或联系作者。

---

**注意**: 本项目仍在开发中，部分功能尚未完成。

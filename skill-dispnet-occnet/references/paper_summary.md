# 论文总结

## 基本信息

**标题**: Unsupervised Light Field Depth Estimation via Multi-view Feature Matching with Occlusion Prediction

**作者**: Shansi Zhang, Nan Meng, Edmund Y. Lam

**发表**: IEEE Transactions on Image Processing

**关键词**: Light field, unsupervised depth estimation, feature matching, occlusion prediction

## 研究动机

### 问题背景
- 光场深度估计是许多应用的基础（自动对焦、场景重建、新视图合成等）
- 学习方法比传统方法精度更高、效率更好
- 但获取足够的深度标签成本很高，尤其是真实世界光场图像
- 现有无监督方法主要适用于密集光场，对稀疏光场（大视差）效果不佳

### 研究目标
开发一个无监督光场深度估计框架，同时适用于：
- 密集光场图像（相邻视图视差范围 [-4, 4]）
- 稀疏光场图像（相邻视图视差范围 [-20, 20]）
- 真实世界光场图像（更好的泛化能力）

## 核心贡献

### 1. DispNet（视差估计网络）
- **Coarse-to-fine 结构**：粗视差估计 + 残差 refinement
- **多视图特征匹配**：显式构建 memory-efficient cost volumes
- **三视图输入策略**：中心视图 + 左右对称源视图
  - 避免遮挡引起的匹配歧义
  - 每个场景点至少在两个视图中可见
- **方差基特征匹配**：比传统串联方式更节省内存

### 2. OccNet（遮挡预测网络）
- **目的**：解决遮挡区域违反 photo-consistency 的问题
- **输入**：warp 后的左右视图 + 预测视差图
- **输出**：左右视图的 confidence maps（遮挡图）
- **作用**：作为 photometric loss 的像素级权重
- **特点**：仅在训练时使用，不影响推理效率

### 3. 视差融合策略
- **多输入组合**：使用不同的视图组合输入 DispNet
- **基于误差的融合**：利用辅助视图估计每个视差图的误差
- **遮挡处理**：
  - 计算多个辅助视图的 warping 误差标准差
  - 标准差 > 阈值 → 遮挡区域 → 使用中值
  - 否则 → 使用均值
- **Weighted fusion**：选择误差最小的 2 个视差进行加权融合

### 4. 实验结果
- 在密集和稀疏光场上都达到 superior performance
- 优于其他无监督方法，接近监督方法
- 在真实世界光场上展现更好的鲁棒性和泛化能力
- 网络轻量（1.802M 参数），推理速度快（2.688s）

## 方法详解

### 光场几何

4D 光场表示：L ∈ R^(U×V×X×Y)
- (u, v)：角度索引（视图位置）
- (x, y)：空间索引（像素位置）
- 由 U×V 个子孔径图像（SAI）组成

中心视差 d(x,y) 与任意视图的关系：
```
L(u_c, v_c, x, y) = L(u, v, x + d(x,y)×(u_c-u), y + d(x,y)×(v_c-v))
```

### 整体框架

**输入策略**：
- 对于 7×7 光场，共 6 种输入组合
- 同一列的视图需要旋转 90°（将垂直视差转为水平）
- 输出视差需要旋转 -90°恢复方向

**处理流程**：
1. 三视图输入 → DispNet
2. 输出多个视差图（不同输入组合）
3. 根据视图距离进行缩放
4. 使用辅助视图（对角线方向）评估误差
5. 基于误差融合得到最终视差图

### DispNet 架构

**特征提取器**：
- 多个 residual blocks（2 个 3×3 conv + leaky ReLU）
- ASPP block（多尺度特征）：
  - 3 个 dilated conv（rate: 3, 6, 8）
  - 1 个 global average pooling

**Coarse 分支**：
- 视差采样：D 个等间距样本 [s_min, ..., s_max]
- 密集光场：[-12, 12]，间隔 1
- 稀疏光场：[-20, 20]，间隔 1.2
- 构建 coarse cost volume（C×D×X×Y）
- Cost filters 处理（3D residual blocks，stride=2 下采样）
- Disparity regression → coarse disparity d̃_coa

**Residual 分支**：
- 残差采样：更小的范围和间隔
- 密集光场：[-1, 1]，间隔 0.1
- 稀疏光场：[-2, 2]，间隔 0.12
- 使用 d̃_coa + s'_i 进行 warp
- 构建 residual cost volume
- 共享 cost filters 处理
- Disparity regression → residual map d̃_res

**最终输出**：
```
d̃ = d̃_coa + d̃_res
```

**视差回归**（soft argmin）：
```
d̃(x,y) = s · softmax(c(x,y))
```
其中 c 是 cost，s 是视差/残差样本向量

### 遮挡预测

**OccNet 结构**：
- U-Net 架构
- Residual blocks + skip connections
- 最大 channel 数：64
- 参数量：仅 0.113M（非常轻量）

**训练流程**：
1. DispNet 输出 d̃
2. 使用 d̃ warp 左右源视图到中心视图：
   ```
   I_l→c(x,y) = I_l(x + d̃(x,y), y)
   I_r→c(x,y) = I_r(x - d̃(x,y), y)
   ```
3. OccNet 输入：[I_l→c, I_r→c, d̃] 拼接
4. OccNet 输出：O_l, O_r（confidence maps）
   - Softmax 激活：O_l(x,y) + O_r(x,y) = 1
5. 重建中心视图：
   ```
   I_rec = O_l ⊙ I_l→c + O_r ⊙ I_r→c
   ```

**物理意义**：
- 遮挡区域 → 大 warping 误差 → 低 confidence
- 非遮挡区域 → 小 warping 误差 → 高 confidence
- 红色像素（值≈1）：对该视图重建贡献大
- 蓝色像素（值≈0）：对该视图重建贡献小

### 损失函数

**完整损失**：
```
ℓ_full = ℓ_wpm + ℓ_rec + α₁ℓ_SSIM + α₂ℓ_smd + α₃ℓ_smo
```

**各项损失详解**：

1. **Weighted Photometric Loss**（ℓ_wpm）：
   ```
   ℓ_wpm = (1/XY) Σ O_l(x,y) ⊙ |I_l→c(x,y) - I_c(x,y)|
         + (1/XY) Σ O_r(x,y) ⊙ |I_r→c(x,y) - I_c(x,y)|
   ```
   - 使用遮挡图作为像素级权重
   - 遮挡区域权重小，减轻不利影响

2. **Reconstruction Loss**（ℓ_rec）：
   ```
   ℓ_rec = ||I_rec - I_c||₁
   ```
   - 训练 OccNet
   - 间接监督 DispNet

3. **SSIM Loss**（ℓ_SSIM）：
   ```
   ℓ_SSIM = 1 - (SSIM(I_l→c, I_c) + SSIM(I_r→c, I_c)) / 2
   ```
   - 增强结构相似性
   - 系数 α₁ = 1

4. **Smoothness Loss**（ℓ_smd）：
   ```
   ℓ_smd = (1/XY) Σ |∇d̃(x,y)| ⊙ exp(-η|∇I_c(x,y)|)
   ```
   - Edge-aware 平滑
   - 在图像边缘处放松平滑约束
   - 系数 α₂ = 0.1，η = 100

5. **Occlusion Smoothness Loss**（ℓ_smo）：
   ```
   ℓ_smo = (1/XY) Σ |∇O_l(x,y)| ⊙ exp(-η|∇I_c(x,y)|)
   ```
   - 平滑遮挡图
   - 仅用于 O_l（因为 O_r = 1 - O_l）
   - 系数 α₃ = 0.05

**注意**：这些损失也应用于 coarse disparity d̃_coa

### 视差融合

**误差估计**：

1. 对于每个视差图 d̂_j 和 Z 个辅助视图：
   - 计算 warping 误差 ε_j ∈ R^(X×Y×Z)

2. 检测遮挡：
   ```
   M_j(x,y) = 1,  if σ_z(ε_j(x,y,z)) > θ(q)
            = 0,  otherwise
   ```
   - σ_z：沿 Z 维度的标准差
   - θ(q)：使用 q 分位数确定阈值
   - q = 0.95（5% 像素使用标准差判断为遮挡）

3. 计算误差图：
   ```
   e_j(x,y) = median_z(ε_j) ⊙ M_j + mean_z(ε_j) ⊙ (1 - M_j)
   ```
   - 遮挡区域使用中值（消除大误差影响）
   - 非遮挡区域使用均值

**融合策略**：

1. **Minimum error fusion**：
   ```
   j' = argmin_j(e_j(x,y))
   d̂_final(x,y) = d̂_j'(x,y)
   ```

2. **Weighted fusion**（推荐）：
   ```
   W(x,y) = softmax(-E(x,y))  # E 是最小的 n' 个误差
   d̂_final(x,y) = Σ w_j(x,y) × d̂_j(x,y)
   ```
   - n' = 2（选择误差最小的 2 个）
   - 实验表明效果最好

## 实验设置

### 数据集

**合成数据**：
- **HCI Dataset**：密集光场，视差范围 [-4, 4]
- **DLF Dataset**：密集光场
- **SLF Dataset**：稀疏光场，视差范围 [-20, 20]

**真实世界**：
- Stanford Lytro LF Archive
- Kalantari Dataset
- EPFL LF Dataset

### 实现细节

**网络配置**：
- DispNet：
  - Feature extractor 最大 channel：128
  - Cost filters 数量：2
  - 总参数量：1.802M
- OccNet：
  - 最大 channel：64
  - 参数量：0.113M

**训练参数**：
- 输入裁剪：256×256（随机）
- 优化器：Adam
- 初始学习率：1e-3
- 衰减：每 50 epoch × 0.8
- Epoch 数：~500
- Batch size：4

**推理配置**：
- 密集光场输入组合：
  - [(u_c-3, v_c), (u_c, v_c), (u_c+3, v_c)]
  - [(u_c-2, v_c), (u_c, v_c), (u_c+2, v_c)]
  - 垂直方向同理，共 6 种
  - Weighted fusion with n'=2
- 稀疏光场输入组合：
  - [(u_c-1, v_c), (u_c, v_c), (u_c+1, v_c)]
  - [(u_c, v_c-1), (u_c, v_c), (u_c, v_c+1)]
  - Minimum error fusion（只有 2 个视差图）

### 主要结果

**HCI Dataset 对比**（Dino 场景）：
- 监督方法：
  - EPINet: MSE=0.167, BPR@0.07=1.286
  - LFAttNet: MSE=0.093, BPR@0.07=0.848
- 无监督方法：
  - UnCNN: MSE=1.807, BPR@0.07=23.660
  - **Ours: MSE=2.266, BPR@0.07=9.238** ✓

**效率对比**：
- 参数量：1.802M（轻量）
- 推理时间：2.688s（NVIDIA Tesla P100）
- 在精度和效率之间取得良好平衡

### 消融实验

**DispNet 设计**：
- w/o coarse-to-fine: MSE=3.057 → Default: MSE=2.266 ✓
- w/o shared weights: MSE=2.282（参数更多但无提升）

**遮挡预测**：
- w/o OccNet: MSE=2.701 → Default: MSE=2.266 ✓
- 验证了 OccNet 的有效性

**损失项**：
- w/o ℓ_SSIM: MSE=2.393
- w/o ℓ_smd: MSE=2.457
- w/o ℓ_smo: MSE=2.276
- Full loss: MSE=2.266 ✓

**融合策略**：
- w/o occlusion handling: MSE=2.837
- Minimum error fusion: MSE=2.411
- Weighted fusion (n'=4): MSE=2.564
- **Weighted fusion (n'=2): MSE=2.266** ✓

## 关键洞察

1. **三视图输入策略**：巧妙解决遮挡问题，每个点至少在两个视图中可见

2. **方差基特征匹配**：
   - 比传统串联方式更节省内存
   - 可适应任意数量输入视图

3. **OccNet 的作用**：
   - 不是简单的后处理
   - 通过 reconstruction loss 联合训练
   - 帮助 DispNet 学习更好的视差

4. **遮挡处理的重要性**：
   - 融合时不处理遮挡：MSE=2.837
   - 使用中值处理遮挡：MSE=2.266
   - 提升显著（20%+）

5. **Coarse-to-fine 设计**：
   - 仅增加 0.003M 参数
   - 但性能提升明显
   - 细采样对精度至关重要

## 局限性与未来方向

### 局限性
- 与监督方法仍有差距（特别是 MSE 指标）
- 对于极端遮挡区域仍有挑战
- 真实世界数据的泛化能力有待进一步提升

### 未来方向
- 探索更强的特征表示
- 改进遮挡检测机制
- 结合传统几何约束
- 扩展到动态场景

## 代码实现要点

### 关键模块
1. **Variance-based feature matching**
2. **Cost volume construction**
3. **3D cost filters**
4. **Soft argmin disparity regression**
5. **Occlusion prediction with U-Net**
6. **Edge-aware smoothness loss**
7. **Disparity fusion with occlusion handling**

### 注意事项
- 视图旋转处理（垂直→水平）
- 视差缩放（根据视图距离）
- 遮挡区域的中值/均值选择
- Quantile q 的选择（q=0.95 最佳）

## 参考链接

- **论文**: 查看原始 PDF
- **HCI Dataset**: http://lightfield-analysis.net/
- **Stanford Lytro**: http://lightfields.stanford.edu/
- **相关代码**: 检查作者是否开源

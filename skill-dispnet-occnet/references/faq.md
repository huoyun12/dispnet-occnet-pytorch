# 常见问题解答 (FAQ)

## 环境配置问题

### Q1: CUDA 版本不匹配

**问题**: 安装 PyTorch 后 CUDA 不可用

**解决方案**:
```bash
# 访问 https://pytorch.org/get-started/locally/ 选择合适的版本
# 或使用以下命令安装带 CUDA 11.8 的版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q2: 内存不足 (OOM)

**问题**: 训练时出现 CUDA out of memory

**解决方案**:
1. **减小 batch size**: 4 → 2 或 1
2. **降低输入分辨率**: 256×256 → 192×192
3. **减少 disparity samples**: 25 → 20
4. **使用梯度累积**:
```python
# 累积 4 个 batch 的梯度
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q3: 3D 卷积内存消耗大

**问题**: Cost volume 处理时内存不足

**解决方案**:
- Cost filters 使用 stride=2 进行下采样
- 减少 cost filters 数量：2 → 1
- 使用更小的 base channels

## 训练问题

### Q4: 损失不收敛

**问题**: 训练损失震荡或不下降

**可能原因**:
1. 学习率过大
2. 数据预处理错误
3. 损失函数实现有误
4. 网络初始化问题

**调试步骤**:
1. **降低学习率**: 1e-3 → 5e-4 或 1e-4
2. **检查输入数据**:
   - 确保视图已正确对齐
   - 检查 warp 操作是否正确
   - 验证归一化范围 (0-1 或 -1~1)
3. **单独测试损失项**:
   - 先只用 ℓ_wpm 和 ℓ_rec
   - 逐步添加其他损失
4. **可视化中间结果**:
   - 检查 cost volume
   - 检查 warp 后的图像
   - 检查遮挡图

### Q5: OccNet 不工作

**问题**: OccNet 预测的遮挡图没有意义

**检查清单**:
- [ ] 输入是否正确拼接：[I_l→c, I_r→c, d̃]
- [ ] Softmax 是否正确应用（O_l + O_r = 1）
- [ ] Reconstruction loss 是否正确计算
- [ ] 是否与 DispNet 联合训练

**调试技巧**:
```python
# 可视化遮挡图
import matplotlib.pyplot as plt

def visualize_occlusion(O_l, O_r):
    """可视化遮挡图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(O_l.cpu().detach().squeeze(), cmap='hot')
    axes[0].set_title('O_l (Left confidence)')
    
    axes[1].imshow(O_r.cpu().detach().squeeze(), cmap='hot')
    axes[1].set_title('O_r (Right confidence)')
    
    # 检查 O_l + O_r = 1
    sum_map = O_l + O_r
    axes[2].imshow(sum_map.cpu().detach().squeeze(), cmap='gray')
    axes[2].set_title(f'Sum (should be 1, actual: {sum_map.mean():.4f})')
    
    plt.savefig('occlusion_visualization.png')
```

### Q6: 深度图质量差

**问题**: 预测深度图有很多噪声或错误

**改进方法**:
1. **增加平滑性损失权重**: α₂ = 0.1 → 0.2
2. **使用更强的 edge-aware**:
   - 增加 η 值：100 → 200
   - 使用图像梯度而不是简单的 exp
3. **后处理**:
```python
# 引导滤波后处理
import cv2

def guided_filter(depth, image, radius=40, eps=1e-6):
    """使用原图作为引导进行滤波"""
    depth_np = depth.cpu().numpy()
    image_np = image.cpu().numpy()
    
    depth_filtered = cv2.ximgproc.guidedFilter(
        guide=image_np,
        src=depth_np,
        radius=radius,
        eps=eps
    )
    return torch.from_numpy(depth_filtered)
```

## 数据问题

### Q7: 光场数据读取

**问题**: 如何读取不同格式的光场数据

**解决方案**:

**HCI Dataset (.png 序列)**:
```python
from PIL import Image
import numpy as np

def load_hci_light_field(base_dir, scene_name):
    """加载 HCI 光场数据"""
    views = []
    for v in range(7):  # 7×7 光场
        for u in range(7):
            view_path = f"{base_dir}/{scene_name}/img{v:02d}{u:02d}.png"
            view = Image.open(view_path)
            views.append(np.array(view))
    
    # 重排为 (V, U, H, W, C)
    views = np.array(views).reshape(7, 7, *views[0].shape)
    return views
```

**Stanford Lytro (.mat 文件)**:
```python
import scipy.io as sio

def load_stanford_light_field(mat_path):
    """加载 Stanford Lytro 光场"""
    data = sio.loadmat(mat_path)
    lf = data['LF']  # (V, U, H, W, C)
    return lf
```

### Q8: 视图旋转处理

**问题**: 垂直方向的视图如何旋转

**解决方案**:
```python
def rotate_view_for_processing(view, angle=90):
    """
    旋转视图以将垂直视差转为水平
    
    Args:
        view: 输入视图 (H, W, C)
        angle: 旋转角度（90 表示逆时针）
    
    Returns:
        rotated_view: 旋转后的视图
    """
    if angle == 90:
        # 逆时针旋转 90 度
        rotated = np.rot90(view, k=1, axes=(0, 1))
    elif angle == -90:
        # 顺时针旋转 90 度
        rotated = np.rot90(view, k=3, axes=(0, 1))
    else:
        rotated = view
    
    return rotated

# 使用示例
# 垂直方向输入需要旋转 90 度
if input_from_vertical:
    view = rotate_view_for_processing(view, angle=90)
    # 处理...
    # 输出视差需要旋转 -90 度恢复
    disparity = rotate_view_for_processing(disparity, angle=-90)
```

## 评估问题

### Q9: 评估指标计算

**问题**: 如何计算 MSE 和 BPR

**代码示例**:
```python
import numpy as np

def compute_metrics(pred_disparity, gt_disparity):
    """
    计算深度估计评估指标
    
    Args:
        pred_disparity: 预测视差图 (H, W)
        gt_disparity: 真实视差图 (H, W)
    
    Returns:
        metrics: 评估指标字典
    """
    # 有效像素掩码
    mask = (gt_disparity > 0) & np.isfinite(gt_disparity)
    
    pred = pred_disparity[mask]
    gt = gt_disparity[mask]
    
    # MSE
    mse = np.mean((pred - gt) ** 2)
    
    # Bad Pixel Ratio (不同阈值)
    thresholds = [0.07, 0.03, 0.01]
    bpr = {}
    for thresh in thresholds:
        bad_pixels = np.abs(pred - gt) > thresh
        bpr[f'bpr_{thresh:.2f}'] = np.mean(bad_pixels) * 100  # 百分比
    
    return {
        'mse': mse,
        **bpr
    }

# 使用示例
metrics = compute_metrics(pred, gt)
print(f"MSE: {metrics['mse']:.4f}")
print(f"BPR@0.07: {metrics['bpr_0.07']:.2f}%")
```

### Q10: 与论文结果对比

**问题**: 复现结果不如论文

**检查清单**:
- [ ] 数据集划分是否一致（train/val/test）
- [ ] 预处理是否相同（裁剪、归一化）
- [ ] 超参数设置是否正确
- [ ] 是否使用相同的评估代码
- [ ] 训练是否充分（500 epochs）
- [ ] 随机种子是否固定

**建议**:
1. 仔细检查论文补充材料
2. 查看作者是否开源代码
3. 在 GitHub Issues 提问
4. 尝试多个随机种子
5. 进行消融实验验证各模块

## 推理问题

### Q11: 视差融合实现

**问题**: 如何实现多视差融合

**代码示例**:
```python
def fuse_disparities(disparities, auxiliary_views, quantile_q=0.95):
    """
    融合多个视差图
    
    Args:
        disparities: 视差图列表 [(H, W), ...]
        auxiliary_views: 辅助视图列表
        quantile_q: 遮挡检测分位数
    
    Returns:
        fused_disparity: 融合后的视差图
    """
    n = len(disparities)
    H, W = disparities[0].shape
    
    # 计算每个视差图的误差
    error_maps = []
    for disp in disparities:
        # 使用辅助视图计算 warping 误差
        warping_errors = []
        for aux_view in auxiliary_views:
            # Warp 辅助视图
            warped = warp_view(aux_view, disp)
            # 计算误差
            error = np.abs(warped - central_view)
            warping_errors.append(error)
        
        warping_errors = np.stack(warping_errors, axis=-1)  # (H, W, Z)
        
        # 检测遮挡
        std_dev = np.std(warping_errors, axis=-1)
        threshold = np.quantile(std_dev, quantile_q)
        occlusion_mask = (std_dev > threshold)
        
        # 计算误差图
        mean_error = np.mean(warping_errors, axis=-1)
        median_error = np.median(warping_errors, axis=-1)
        
        error_map = median_error * occlusion_mask + mean_error * (1 - occlusion_mask)
        error_maps.append(error_map)
    
    # Weighted fusion (选择误差最小的 2 个)
    error_maps = np.stack(error_maps, axis=-1)  # (H, W, n)
    disparities = np.stack(disparities, axis=-1)  # (H, W, n)
    
    # 对每个像素，选择误差最小的 2 个
    sorted_indices = np.argsort(error_maps, axis=-1)
    best_2_indices = sorted_indices[:, :, :2]
    
    # 获取对应的误差
    best_2_errors = np.take_along_axis(error_maps, best_2_indices, axis=-1)
    best_2_disparities = np.take_along_axis(disparities, best_2_indices, axis=-1)
    
    # Softmax 权重
    weights = np.exp(-best_2_errors)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    
    # 加权融合
    fused_disparity = np.sum(best_2_disparities * weights, axis=-1)
    
    return fused_disparity
```

## 性能优化

### Q12: 训练速度慢

**问题**: 训练速度太慢

**优化建议**:
1. **使用 cudnn benchmark**:
```python
torch.backends.cudnn.benchmark = True
```

2. **数据预加载**:
```python
dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=8,  # 增加 worker 数量
    pin_memory=True,
    prefetch_factor=2  # 预加载 batch
)
```

3. **混合精度训练**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Q13: 推理速度慢

**问题**: 推理时间超过 2.688s

**优化建议**:
1. **减少输入组合数量**: 6 → 4
2. **使用更小的网络**: 减少 cost filters
3. **TensorRT 加速**: 导出为 ONNX 并使用 TensorRT
4. **多 GPU 并行**: 不同输入组合在不同 GPU 上处理

## 其他资源

- **PyTorch 论坛**: https://discuss.pytorch.org/
- **Stack Overflow**: https://stackoverflow.com/questions/tagged/pytorch
- **GitHub Issues**: 查看相关项目的 issue
- **Papers With Code**: https://paperswithcode.com/
- **光场社区**: http://lightfield-analysis.net/

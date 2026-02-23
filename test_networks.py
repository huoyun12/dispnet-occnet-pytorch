"""
测试 DispNet 和 OccNet 网络
"""

import sys
import torch

print("="*60)
print("DispNet-OccNet 网络测试")
print("="*60)

# 测试环境
print("\n1. 环境信息:")
print(f"  Python 版本：{sys.version}")
print(f"  PyTorch 版本：{torch.__version__}")
print(f"  CUDA 可用：{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 版本：{torch.version.cuda}")
    print(f"  GPU 数量：{torch.cuda.device_count()}")

# 测试 DispNet
print("\n2. 测试 DispNet:")
try:
    from src.dispnet import build_dispnet
    
    config = {
        'input_channels': 3,
        'base_channels': 64,
        'max_channels': 128,
        'coarse_min': -12,
        'coarse_max': 12,
        'coarse_samples': 25,
        'residual_min': -1,
        'residual_max': 1,
        'residual_samples': 21
    }
    
    model = build_dispnet(config)
    
    # 创建测试输入
    batch_size = 2
    views = [
        torch.randn(batch_size, 3, 256, 256),
        torch.randn(batch_size, 3, 256, 256),
        torch.randn(batch_size, 3, 256, 256)
    ]
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(views)
    
    print(f"  ✓ DispNet 创建成功")
    print(f"  输入：{[v.shape for v in views]}")
    print(f"  输出:")
    print(f"    disparity_coarse: {outputs['disparity_coarse'].shape}")
    print(f"    disparity_residual: {outputs['disparity_residual'].shape}")
    print(f"    disparity: {outputs['disparity'].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量：{total_params:,} ({total_params/1e6:.3f}M)")
    
except Exception as e:
    print(f"  ✗ DispNet 测试失败：{e}")
    import traceback
    traceback.print_exc()

# 测试 OccNet
print("\n3. 测试 OccNet:")
try:
    from src.occnet import build_occnet
    
    model = build_occnet()
    
    # 创建测试输入
    batch_size = 2
    warped_left = torch.randn(batch_size, 3, 256, 256)
    warped_right = torch.randn(batch_size, 3, 256, 256)
    disparity = torch.randn(batch_size, 1, 256, 256)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(warped_left, warped_right, disparity)
    
    print(f"  ✓ OccNet 创建成功")
    print(f"  输入:")
    print(f"    warped_left: {warped_left.shape}")
    print(f"    warped_right: {warped_right.shape}")
    print(f"    disparity: {disparity.shape}")
    print(f"  输出:")
    print(f"    occlusion_left: {outputs['occlusion_left'].shape}")
    print(f"    occlusion_right: {outputs['occlusion_right'].shape}")
    print(f"    occlusion_maps: {outputs['occlusion_maps'].shape}")
    
    # 验证 softmax: O_l + O_r = 1
    occ_sum = outputs['occlusion_left'] + outputs['occlusion_right']
    print(f"  验证：O_l + O_r = {occ_sum.mean().item():.6f} (应该接近 1.0)")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量：{total_params:,} ({total_params/1e6:.3f}M)")
    
except Exception as e:
    print(f"  ✗ OccNet 测试失败：{e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("测试完成!")
print("="*60)

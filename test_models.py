import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import sys

print("=" * 60)
print("DispNet-OccNet 快速验证脚本")
print("=" * 60)

print("\n[1/5] 导入模块...")

try:
    from models.dispnet import DispNet
    from models.occnet import OccNet
    from utils.losses import FullLoss, warp_image
    from data.lightfield_dataset import LightFieldDataset, get_dataloader
    print("  ✓ 所有模块导入成功")
except ImportError as e:
    print(f"  ✗ 模块导入失败: {e}")
    sys.exit(1)

print("\n[2/5] 测试 DispNet 模型...")

try:
    dispnet = DispNet(
        in_channels=3,
        max_channels=128,
        num_disparities=25,
        num_residual_disparities=21,
        num_cost_filters=2
    )
    dispnet_params = sum(p.numel() for p in dispnet.parameters())
    print(f"  ✓ DispNet 创建成功")
    print(f"    参数数量: {dispnet_params / 1e6:.3f}M")
except Exception as e:
    print(f"  ✗ DispNet 创建失败: {e}")
    sys.exit(1)

print("\n[3/5] 测试 OccNet 模型...")

try:
    occnet = OccNet(in_channels=7, max_channels=64)
    occnet_params = sum(p.numel() for p in occnet.parameters())
    print(f"  ✓ OccNet 创建成功")
    print(f"    参数数量: {occnet_params / 1e6:.3f}M")
except Exception as e:
    print(f"  ✗ OccNet 创建失败: {e}")
    sys.exit(1)

print("\n[4/5] 测试前向传播...")

try:
    B, C, H, W = 2, 3, 256, 256

    img_center = torch.randn(B, C, H, W)
    img_left = torch.randn(B, C, H, W)
    img_right = torch.randn(B, C, H, W)

    print(f"  输入形状: center={img_center.shape}, left={img_left.shape}, right={img_right.shape}")

    coarse_d, refined_d, features = dispnet(img_center, img_left, img_right)

    print(f"  ✓ DispNet 前向传播成功")
    print(f"    粗视差形状: {coarse_d.shape}")
    print(f"    精炼视差形状: {refined_d.shape}")
    print(f"    特征形状: {features.shape}")

    warped_left = warp_image(img_left, refined_d)
    warped_right = warp_image(img_right, -refined_d)

    print(f"  [DEBUG] warped_left: {warped_left.shape}, warped_right: {warped_right.shape}, refined_d: {refined_d.shape}")

    refined_d_4d = refined_d.unsqueeze(1) if refined_d.dim() == 3 else refined_d
    print(f"  [DEBUG] refined_d_4d before interpolate: {refined_d_4d.shape}")

    if refined_d_4d.shape[2] != warped_left.shape[2] or refined_d_4d.shape[3] != warped_left.shape[3]:
        refined_d_4d = F.interpolate(refined_d_4d, size=warped_left.shape[2:], mode='bilinear', align_corners=False)
        print(f"  [DEBUG] refined_d_4d after interpolate: {refined_d_4d.shape}")

    occ_input = torch.cat([warped_left, warped_right, refined_d_4d], dim=1)
    print(f"  [DEBUG] occ_input: {occ_input.shape}")
    occ_left, occ_right = occnet(occ_input)

    print(f"  ✓ OccNet 前向传播成功")
    print(f"    遮挡图(左)形状: {occ_left.shape}")
    print(f"    遮挡图(右)形状: {occ_right.shape}")

except Exception as e:
    print(f"  ✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[5/5] 测试损失函数...")

try:
    criterion = FullLoss(alpha_ssim=1.0, alpha_smd=0.1, alpha_smo=0.05, eta=100.0)

    loss, loss_dict = criterion(
        img_center, img_left, img_right,
        coarse_d, refined_d,
        occ_left, occ_right
    )

    print(f"  ✓ 损失计算成功")
    print(f"    总损失: {loss_dict['total']:.4f}")
    print(f"    WPM损失: {loss_dict['wpm']:.4f}")
    print(f"    REC损失: {loss_dict['rec']:.4f}")
    print(f"    SSIM损失: {loss_dict['ssim']:.4f}")
    print(f"    SMD损失: {loss_dict['smd']:.4f}")
    print(f"    SMO损失: {loss_dict['smo']:.4f}")

except Exception as e:
    print(f"  ✗ 损失计算失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有验证测试通过！")
print("=" * 60)

print("\n模型参数统计:")
print(f"  DispNet: {dispnet_params / 1e6:.3f}M")
print(f"  OccNet:  {occnet_params / 1e6:.3f}M")
print(f"  总计:    {(dispnet_params + occnet_params) / 1e6:.3f}M")
print(f"\n论文参考: DispNet=1.802M, OccNet=0.113M, 总计=1.915M")

print("\n下一步:")
print("  1. 准备数据集 (参考 scripts/prepare_data.py)")
print("  2. 运行训练: python train.py --data_dir ./data/hci/train")
print("  3. 运行推理: python inference.py --model_path checkpoints/best_model.pth --data_dir ./data/hci/test")

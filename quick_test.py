#!/usr/bin/env python3
"""
快速验证测试

验证数据集加载和网络前向传播
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.data import HCIDataset
from src.dispnet import build_dispnet
from src.occnet import build_occnet
from src.loss import build_loss
from src.data import warp_view


def main():
    print("="*60)
    print("DispNet-OccNet 快速验证测试")
    print("="*60)
    
    # 1. 测试数据集
    print("\n1. 测试 HCI 数据集...")
    data_dir = Path(r"E:\光场相机相关\深度估计\深度学习\Dispnet-Occnet\data\HCI_dataset")
    
    if not data_dir.exists():
        print(f"✗ 数据目录不存在")
        return False
    
    # 扫描场景
    scenes = [d.name for d in data_dir.iterdir() if d.is_dir() and (d / 'input_Cam000.png').exists()]
    print(f"  找到场景：{len(scenes)} 个")
    
    if len(scenes) == 0:
        print("✗ 未找到有效场景")
        return False
    
    # 测试第一个场景
    test_scene = scenes[0]
    print(f"  测试场景：{test_scene}")
    
    try:
        dataset = HCIDataset(
            data_dir=str(data_dir / test_scene),
            split='train',
            augment=False,
            crop_size=(256, 256)
        )
        
        print(f"  ✓ 数据集创建成功")
        print(f"    样本数：{len(dataset)}")
        
        if len(dataset) == 0:
            print("  ✗ 数据集中没有样本")
            return False
        
        # 加载样本
        sample = dataset[0]
        print(f"  ✓ 成功加载样本")
        print(f"    视图数：{sample['views'].shape[0]}")
        print(f"    视图形状：{sample['views'].shape[1:]}")
        
    except Exception as e:
        print(f"  ✗ 数据集加载失败：{e}")
        return False
    
    # 2. 测试网络
    print("\n2. 测试网络前向传播...")
    
    views = sample['views'].unsqueeze(0)  # [1, N, C, H, W]
    n_views = views.size(1)
    center_idx = n_views // 2
    
    if n_views < 3:
        print(f"  ✗ 视图数量不足：{n_views}")
        return False
    
    I_left = views[:, center_idx - 1]
    I_center = views[:, center_idx]
    I_right = views[:, center_idx + 1]
    views_list = [I_left, I_center, I_right]
    
    # DispNet
    print("  测试 DispNet...")
    try:
        config_disp = {
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
        
        model_disp = build_dispnet(config_disp)
        model_disp.eval()
        
        with torch.no_grad():
            disp_outputs = model_disp(views_list)
        
        print(f"    ✓ DispNet 成功")
        print(f"      参数量：{sum(p.numel() for p in model_disp.parameters()):,}")
        print(f"      输出：{disp_outputs['disparity'].shape}")
        
    except Exception as e:
        print(f"    ✗ DispNet 失败：{e}")
        return False
    
    # OccNet
    print("  测试 OccNet...")
    try:
        disparity = disp_outputs['disparity']
        I_left_warp = warp_view(I_left, disparity, direction='horizontal')
        I_right_warp = warp_view(I_right, disparity, direction='horizontal')
        
        config_occ = {
            'input_channels': 7,
            'base_channels': 32,
            'max_channels': 64
        }
        
        model_occ = build_occnet(config_occ)
        model_occ.eval()
        
        with torch.no_grad():
            occ_outputs = model_occ(I_left_warp, I_right_warp, disparity)
        
        occ_sum = (occ_outputs['occlusion_left'] + occ_outputs['occlusion_right']).mean().item()
        print(f"    ✓ OccNet 成功")
        print(f"      参数量：{sum(p.numel() for p in model_occ.parameters()):,}")
        print(f"      验证：O_l + O_r = {occ_sum:.4f}")
        
    except Exception as e:
        print(f"    ✗ OccNet 失败：{e}")
        return False
    
    # 损失函数
    print("  测试损失函数...")
    try:
        warped_views = {'left': I_left_warp, 'right': I_right_warp}
        
        config_loss = {
            'lambda_wpm': 1.0,
            'lambda_rec': 1.0,
            'lambda_ssim': 1.0,
            'lambda_smd': 0.1,
            'lambda_smo': 0.05
        }
        
        criterion = build_loss(config_loss)
        
        with torch.no_grad():
            loss_data = criterion(disp_outputs, occ_outputs, views_list, warped_views)
        
        print(f"    ✓ 损失计算成功")
        print(f"      total: {loss_data['total'].item():.6f}")
        
    except Exception as e:
        print(f"    ✗ 损失函数失败：{e}")
        return False
    
    # 成功
    print("\n" + "="*60)
    print("✓✓✓ 所有测试通过！✓✓✓")
    print("="*60)
    print("\n可以开始训练:")
    print("  python train.py --config configs/dense_lf.yaml --data_dir <数据路径>")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

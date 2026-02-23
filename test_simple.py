"""简单测试脚本"""

import sys
sys.path.insert(0, '.')

print("测试 DispNet...")
try:
    from src.dispnet import build_dispnet
    model = build_dispnet()
    print("✓ DispNet 创建成功")
    
    import torch
    views = [torch.randn(2, 3, 256, 256) for _ in range(3)]
    model.eval()
    with torch.no_grad():
        outputs = model(views)
    print(f"✓ DispNet 前向传播成功")
    print(f"  输出形状：{outputs['disparity'].shape}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量：{params:,} ({params/1e6:.3f}M)")
    
except Exception as e:
    print(f"✗ DispNet 失败：{e}")
    import traceback
    traceback.print_exc()

print("\n测试 OccNet...")
try:
    from src.occnet import build_occnet
    model = build_occnet()
    print("✓ OccNet 创建成功")
    
    import torch
    warped_left = torch.randn(2, 3, 256, 256)
    warped_right = torch.randn(2, 3, 256, 256)
    disparity = torch.randn(2, 1, 256, 256)
    
    model.eval()
    with torch.no_grad():
        outputs = model(warped_left, warped_right, disparity)
    print(f"✓ OccNet 前向传播成功")
    print(f"  输出形状：{outputs['occlusion_left'].shape}")
    
    # 验证 O_l + O_r = 1
    occ_sum = (outputs['occlusion_left'] + outputs['occlusion_right']).mean().item()
    print(f"  验证 O_l + O_r = {occ_sum:.6f}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量：{params:,} ({params/1e6:.3f}M)")
    
except Exception as e:
    print(f"✗ OccNet 失败：{e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")

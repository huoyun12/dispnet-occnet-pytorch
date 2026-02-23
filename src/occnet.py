"""
OccNet: 遮挡预测网络

U-Net 结构，预测左右视图的 confidence maps，用作 photometric loss 的像素级权重。
仅在训练时使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    """Convolutional block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv = ConvBlock(out_channels * 2, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Pad if necessary
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class OccNet(nn.Module):
    """
    OccNet: 遮挡预测网络
    
    U-Net 架构，预测左右视图的 confidence maps。
    
    输入：[I_l→c, I_r→c, d̃] 拼接 (7 channels)
    输出：[O_l, O_r] confidence maps (2 channels, softmax)
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        if config is None:
            config = {
                'input_channels': 7,  # I_l→c (3) + I_r→c (3) + disparity (1)
                'base_channels': 32,
                'max_channels': 64,
                'num_levels': 4
            }
        
        self.config = config
        base_ch = config.get('base_channels', 32)
        max_ch = config.get('max_channels', 64)
        in_ch = config.get('input_channels', 7)
        
        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, max_ch)
        self.enc4 = DownBlock(max_ch, max_ch)
        
        # Bottleneck
        self.bottleneck = ConvBlock(max_ch, max_ch)
        
        # Decoder
        self.dec4 = UpBlock(max_ch, max_ch)
        self.dec3 = UpBlock(max_ch, base_ch * 2)
        self.dec2 = UpBlock(base_ch * 2, base_ch)
        
        # Output
        self.out_conv = nn.Conv2d(base_ch, 2, 1)
        
        # Softmax for confidence maps
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, warped_left: torch.Tensor, warped_right: torch.Tensor,
                disparity: torch.Tensor) -> dict:
        """
        前向传播
        
        Args:
            warped_left: Warp 后的左视图 I_l→c [B, 3, H, W]
            warped_right: Warp 后的右视图 I_r→c [B, 3, H, W]
            disparity: 预测的视差图 [B, 1, H, W]
        
        Returns:
            outputs: 输出字典
                - 'occlusion_left': 左视图遮挡图 O_l [B, 1, H, W]
                - 'occlusion_right': 右视图遮挡图 O_r [B, 1, H, W]
                - 'occlusion_maps': 原始遮挡图 [B, 2, H, W]
        """
        # 拼接输入
        x = torch.cat([warped_left, warped_right, disparity], dim=1)  # [B, 7, H, W]
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # Decoder
        dec4 = self.dec4(bottleneck, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        
        # Output
        occ_maps = self.out_conv(dec2)  # [B, 2, H, W]
        
        # Softmax 获取 confidence maps
        occ_maps_softmax = self.softmax(occ_maps)  # [B, 2, H, W]
        
        # 分离左右遮挡图
        occ_left = occ_maps_softmax[:, 0:1, :, :]  # O_l
        occ_right = occ_maps_softmax[:, 1:2, :, :]  # O_r
        
        return {
            'occlusion_left': occ_left,
            'occlusion_right': occ_right,
            'occlusion_maps': occ_maps_softmax
        }


def build_occnet(config: dict = None) -> OccNet:
    """构建 OccNet"""
    return OccNet(config)


if __name__ == '__main__':
    # 测试 OccNet
    model = build_occnet()
    
    # 创建测试输入
    batch_size = 2
    warped_left = torch.randn(batch_size, 3, 256, 256)
    warped_right = torch.randn(batch_size, 3, 256, 256)
    disparity = torch.randn(batch_size, 1, 256, 256)
    
    # 前向传播
    outputs = model(warped_left, warped_right, disparity)
    
    print("OccNet 测试:")
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
    print(f"  O_l + O_r = {occ_sum.mean().item():.6f} (应该接近 1.0)")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量：{total_params:,} ({total_params/1e6:.3f}M)")

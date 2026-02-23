"""
DispNet: 视差估计网络

采用 coarse-to-fine 结构，通过多视图特征匹配学习视差。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ResidualBlock(nn.Module):
    """Residual block with 2 convolutions"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class ASPPBlock(nn.Module):
    """Atrous Spatial Pyramid Pooling block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 3, dilation=3, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, 8, dilation=8, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat4 = F.interpolate(feat4, size=(h, w), mode='bilinear', align_corners=False)
        
        out = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class FeatureExtractor(nn.Module):
    """特征提取器"""
    
    def __init__(self, input_channels: int = 3, base_channels: int = 64, max_channels: int = 128):
        super().__init__()
        
        # 初始卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, max_channels, 2, stride=2)
        
        # ASPP
        self.aspp = ASPPBlock(max_channels, max_channels // 4)
        
        self.out_channels = max_channels + max_channels  # max_channels + aspp output
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # ASPP
        aspp_feat = self.aspp(x)
        x = torch.cat([x, aspp_feat], dim=1)
        
        return x


class VarianceCostVolume(nn.Module):
    """基于方差的特征匹配构建代价体"""
    
    def __init__(self, max_disparity: int = 25):
        super().__init__()
        self.max_disparity = max_disparity
    
    def forward(self, ref_feat: torch.Tensor, src_feats: List[torch.Tensor], 
                disparities: torch.Tensor) -> torch.Tensor:
        """
        构建方差基代价体
        
        Args:
            ref_feat: 参考视图特征 [B, C, H, W]
            src_feats: 源视图特征列表，每个 [B, C, H, W]
            disparities: 视差采样 [D]
        
        Returns:
            cost_volume: 代价体 [B, C, D, H, W]
        """
        b, c, h, w = ref_feat.shape
        d = len(disparities)
        
        # 初始化代价体
        cost_volume = torch.zeros(b, c, d, h, w, device=ref_feat.device)
        
        # 对每个源视图
        for src_feat in src_feats:
            # 对每个视差
            for i, disp in enumerate(disparities):
                # Warp 源特征到参考视图
                warped = self.warp_feature(src_feat, disp)
                
                # 计算方差（特征匹配代价）
                diff = ref_feat - warped
                cost = diff ** 2
                
                # 累加
                cost_volume[:, :, i, :, :] += cost
        
        # 平均
        cost_volume /= len(src_feats)
        
        return cost_volume
    
    def warp_feature(self, feat: torch.Tensor, disparity: float) -> torch.Tensor:
        """Warp 特征图"""
        b, c, h, w = feat.shape
        
        # 生成采样网格
        grid = torch.meshgrid(
            torch.arange(h, device=feat.device),
            torch.arange(w, device=feat.device),
            indexing='ij'
        )
        grid = torch.stack(grid, dim=-1).unsqueeze(0).float()  # [1, H, W, 2]
        
        # 应用视差偏移（假设水平视差）
        grid[:, :, :, 0] = grid[:, :, :, 0] - disparity
        
        # 归一化到 [-1, 1]
        grid[:, :, :, 0] = grid[:, :, :, 0] / (h - 1) * 2 - 1
        grid[:, :, :, 1] = grid[:, :, :, 1] / (w - 1) * 2 - 1
        
        # Grid sample
        warped = F.grid_sample(
            feat, grid.expand(b, -1, -1, -1),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped


class CostFilter3D(nn.Module):
    """3D cost filter"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class DisparityRegression(nn.Module):
    """Soft argmin 视差回归"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, cost_volume: torch.Tensor, disparities: torch.Tensor) -> torch.Tensor:
        """
        Soft argmin 视差回归
        
        Args:
            cost_volume: 代价体 [B, C, D, H, W]
            disparities: 视差采样 [D]
        
        Returns:
            disparity_map: 预测视差图 [B, 1, H, W]
        """
        # 沿 D 维度求平均得到 cost
        cost = cost_volume.mean(dim=1, keepdim=False)  # [B, D, H, W]
        
        # Softmax 获取概率
        prob = F.softmax(cost, dim=1)  # [B, D, H, W]
        
        # 期望值
        disparity_map = torch.sum(
            prob * disparities.view(1, -1, 1, 1),
            dim=1,
            keepdim=True
        )  # [B, 1, H, W]
        
        return disparity_map


class DispNet(nn.Module):
    """
    DispNet: 视差估计网络
    
    Coarse-to-fine 结构：
    1. Coarse 分支：估计粗略视差
    2. Residual 分支：估计残差视差
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            input_channels=config.get('input_channels', 3),
            base_channels=config.get('base_channels', 64),
            max_channels=config.get('max_channels', 128)
        )
        
        # 代价体构建
        self.cost_volume = VarianceCostVolume(
            max_disparity=config.get('max_disparity', 25)
        )
        
        # Cost filters (coarse)
        self.coarse_filters = nn.Sequential(
            CostFilter3D(config.get('max_channels', 128), 64, stride=1),
            CostFilter3D(64, 32, stride=2),
            CostFilter3D(32, 16, stride=2)
        )
        
        # Coarse disparity regression
        self.coarse_regression = DisparityRegression()
        
        # Residual branch
        self.residual_filters = nn.Sequential(
            CostFilter3D(config.get('max_channels', 128), 64, stride=1),
            CostFilter3D(64, 32, stride=2),
            CostFilter3D(32, 16, stride=2)
        )
        
        # Residual disparity regression
        self.residual_regression = DisparityRegression()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, views: List[torch.Tensor]) -> dict:
        """
        前向传播
        
        Args:
            views: 输入视图列表 [I_left, I_center, I_right]，每个 [B, 3, H, W]
        
        Returns:
            outputs: 输出字典
                - 'disparity_coarse': 粗略视差图 [B, 1, H, W]
                - 'disparity_residual': 残差视差图 [B, 1, H, W]
                - 'disparity': 最终视差图 [B, 1, H, W]
        """
        # 提取特征
        ref_view = views[1]  # 中心视图作为参考
        src_views = [views[0], views[2]]  # 左右视图作为源
        
        ref_feat = self.feature_extractor(ref_view)
        src_feats = [self.feature_extractor(v) for v in src_views]
        
        # Coarse disparity sampling
        coarse_disparities = torch.linspace(
            self.config.get('coarse_min', -12),
            self.config.get('coarse_max', 12),
            self.config.get('coarse_samples', 25)
        ).to(ref_feat.device)
        
        # 构建 coarse cost volume
        coarse_cost = self.cost_volume(ref_feat, src_feats, coarse_disparities)
        
        # Cost filtering
        coarse_cost_filtered = self.coarse_filters(coarse_cost)
        
        # Coarse disparity regression
        disparity_coarse = self.coarse_regression(coarse_cost_filtered, coarse_disparities)
        
        # Residual disparity sampling
        residual_disparities = torch.linspace(
            self.config.get('residual_min', -1),
            self.config.get('residual_max', 1),
            self.config.get('residual_samples', 21)
        ).to(ref_feat.device)
        
        # Warp source features using coarse disparity
        warped_src_feats = []
        for src_feat in src_feats:
            warped = self.warp_feature(src_feat, disparity_coarse)
            warped_src_feats.append(warped)
        
        # 构建 residual cost volume
        residual_cost = self.cost_volume(ref_feat, warped_src_feats, residual_disparities)
        
        # Cost filtering
        residual_cost_filtered = self.residual_filters(residual_cost)
        
        # Residual disparity regression
        disparity_residual = self.residual_regression(residual_cost_filtered, residual_disparities)
        
        # Final disparity
        disparity_final = disparity_coarse + disparity_residual
        
        return {
            'disparity_coarse': disparity_coarse,
            'disparity_residual': disparity_residual,
            'disparity': disparity_final
        }
    
    def warp_feature(self, feat: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
        """Warp 特征图使用预测的视差"""
        b, c, h, w = feat.shape
        
        # 生成采样网格
        grid = torch.meshgrid(
            torch.arange(h, device=feat.device),
            torch.arange(w, device=feat.device),
            indexing='ij'
        )
        grid = torch.stack(grid, dim=-1).unsqueeze(0).float()  # [1, H, W, 2]
        
        # 应用视差偏移
        grid[:, :, :, 0] = grid[:, :, :, 0] - disparity.squeeze(1)
        
        # 归一化到 [-1, 1]
        grid[:, :, :, 0] = grid[:, :, :, 0] / (h - 1) * 2 - 1
        grid[:, :, :, 1] = grid[:, :, :, 1] / (w - 1) * 2 - 1
        
        # Grid sample
        warped = F.grid_sample(
            feat, grid.expand(b, -1, -1, -1),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped


def build_dispnet(config: dict = None) -> DispNet:
    """构建 DispNet"""
    if config is None:
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
    
    return DispNet(config)


if __name__ == '__main__':
    # 测试 DispNet
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
    outputs = model(views)
    
    print("DispNet 测试:")
    print(f"  输入: {[v.shape for v in views]}")
    print(f"  输出:")
    print(f"    disparity_coarse: {outputs['disparity_coarse'].shape}")
    print(f"    disparity_residual: {outputs['disparity_residual'].shape}")
    print(f"    disparity: {outputs['disparity'].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量：{total_params:,} ({total_params/1e6:.3f}M)")

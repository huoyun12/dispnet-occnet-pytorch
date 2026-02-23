"""
损失函数模块

实现论文中的无监督损失函数：
1. Weighted Photometric Loss (ℓ_wpm)
2. Reconstruction Loss (ℓ_rec)
3. SSIM Loss (ℓ_SSIM)
4. Smoothness Loss (ℓ_smd)
5. Occlusion Smoothness Loss (ℓ_smo)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PhotometricLoss(nn.Module):
    """光度一致性损失"""
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        计算光度损失
        
        Args:
            target: 目标视图 [B, C, H, W]
            source: 源视图 [B, C, H, W]
        
        Returns:
            loss: 光度损失
        """
        if self.loss_type == 'l1':
            return F.l1_loss(target, source, reduction='none')
        elif self.loss_type == 'l2':
            return F.mse_loss(target, source, reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")


class SSIMLoss(nn.Module):
    """
    SSIM (Structural Similarity) 损失
    
    使用高斯窗口计算局部 SSIM
    """
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 1
        self.window = self._create_window(window_size, sigma)
    
    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """创建 1D 高斯窗口"""
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size]
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        计算 SSIM 损失
        
        Args:
            img1: 图像 1 [B, C, H, W]
            img2: 图像 2 [B, C, H, W]
        
        Returns:
            ssim_loss: 1 - SSIM
        """
        C = img1.shape[1]
        
        # 创建 2D 高斯窗口
        window_1d = self.window.to(img1.device)
        window_2d = window_1d.transpose(2, 3) @ window_1d  # [1, 1, window_size, window_size]
        window = window_2d.expand(C, 1, -1, -1).contiguous()
        
        # 计算均值
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=C)
        
        # 计算方差和协方差
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=C) - mu1_mu2
        
        # SSIM 参数
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 计算 SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # 返回 1 - SSIM 作为损失
        return 1 - ssim_map.mean()


class SmoothnessLoss(nn.Module):
    """
    Edge-aware 平滑性损失
    
    鼓励视差图平滑，但在图像边缘处放松约束
    """
    
    def __init__(self, edge_aware: bool = True, eta: float = 100):
        super().__init__()
        self.edge_aware = edge_aware
        self.eta = eta
    
    def forward(self, disparity: torch.Tensor, 
                image: torch.Tensor = None) -> torch.Tensor:
        """
        计算平滑性损失
        
        Args:
            disparity: 视差图 [B, 1, H, W]
            image: 参考图像（用于 edge-aware）[B, C, H, W]
        
        Returns:
            loss: 平滑性损失
        """
        # 计算视差梯度
        grad_x = disparity[:, :, :, :-1] - disparity[:, :, :, 1:]
        grad_y = disparity[:, :, :-1, :] - disparity[:, :, 1:, :]
        
        if self.edge_aware and image is not None:
            # 计算图像梯度
            img_grad_x = image[:, :, :, :-1] - image[:, :, :, 1:]
            img_grad_y = image[:, :, :-1, :] - image[:, :, 1:, :]
            
            # 计算 edge-aware 权重
            weight_x = torch.exp(-self.eta * torch.mean(torch.abs(img_grad_x), dim=1, keepdim=True))
            weight_y = torch.exp(-self.eta * torch.mean(torch.abs(img_grad_y), dim=1, keepdim=True))
            
            # 加权梯度
            loss = torch.mean(torch.abs(grad_x * weight_x)) + \
                   torch.mean(torch.abs(grad_y * weight_y))
        else:
            loss = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))
        
        return loss


class UnsupervisedLoss(nn.Module):
    """
    无监督光场深度估计总损失
    
    ℓ_full = ℓ_wpm + ℓ_rec + α₁ℓ_SSIM + α₂ℓ_smd + α₃ℓ_smo
    
    其中：
    - ℓ_wpm: Weighted Photometric Loss
    - ℓ_rec: Reconstruction Loss
    - ℓ_SSIM: SSIM Loss
    - ℓ_smd: Disparity Smoothness Loss
    - ℓ_smo: Occlusion Smoothness Loss
    """
    
    def __init__(self, config: dict = None):
        super().__init__()
        
        if config is None:
            config = {
                'lambda_wpm': 1.0,      # Weighted Photometric Loss 权重
                'lambda_rec': 1.0,      # Reconstruction Loss 权重
                'lambda_ssim': 1.0,     # SSIM Loss 权重 (α₁)
                'lambda_smd': 0.1,      # Disparity Smoothness 权重 (α₂)
                'lambda_smo': 0.05,     # Occlusion Smoothness 权重 (α₃)
                'photo_loss_type': 'l1',
                'ssim_window_size': 11,
                'smoothness_eta': 100
            }
        
        self.config = config
        
        # 初始化各个损失
        self.photo_loss = PhotometricLoss(config.get('photo_loss_type', 'l1'))
        self.ssim_loss = SSIMLoss(config.get('ssim_window_size', 11))
        self.smoothness_loss = SmoothnessLoss(
            edge_aware=True,
            eta=config.get('smoothness_eta', 100)
        )
        
        # 损失权重
        self.lambda_wpm = config.get('lambda_wpm', 1.0)
        self.lambda_rec = config.get('lambda_rec', 1.0)
        self.lambda_ssim = config.get('lambda_ssim', 1.0)
        self.lambda_smd = config.get('lambda_smd', 0.1)
        self.lambda_smo = config.get('lambda_smo', 0.05)
    
    def forward(self, disp_outputs: Dict, occ_outputs: Dict,
                views: list, warped_views: Dict) -> Dict:
        """
        计算总损失
        
        Args:
            disp_outputs: DispNet 输出
                - 'disparity': 最终视差图 [B, 1, H, W]
                - 'disparity_coarse': 粗略视差图 [B, 1, H, W]
            occ_outputs: OccNet 输出
                - 'occlusion_left': 左视图遮挡图 [B, 1, H, W]
                - 'occlusion_right': 右视图遮挡图 [B, 1, H, W]
                - 'occlusion_maps': 原始遮挡图 [B, 2, H, W]
            views: 输入视图列表 [I_left, I_center, I_right]
            warped_views: Warp 后的视图
                - 'left': I_l→c [B, 3, H, W]
                - 'right': I_r→c [B, 3, H, W]
        
        Returns:
            loss_dict: 损失字典
                - 'total': 总损失
                - 'wpm': Weighted Photometric Loss
                - 'rec': Reconstruction Loss
                - 'ssim': SSIM Loss
                - 'smd': Disparity Smoothness Loss
                - 'smo': Occlusion Smoothness Loss
        """
        loss_dict = {}
        
        # 提取数据
        I_c = views[1]  # 中心视图
        I_l_warp = warped_views['left']  # Warp 后的左视图
        I_r_warp = warped_views['right']  # Warp 后的右视图
        
        O_l = occ_outputs['occlusion_left']  # 左视图遮挡图
        O_r = occ_outputs['occlusion_right']  # 右视图遮挡图
        
        d̃ = disp_outputs['disparity']  # 最终视差
        d̃_coarse = disp_outputs.get('disparity_coarse', d̃)  # 粗略视差
        
        # 1. Weighted Photometric Loss (ℓ_wpm)
        # ℓ_wpm = (1/XY) Σ O_l ⊙ |I_l→c - I_c| + (1/XY) Σ O_r ⊙ |I_r→c - I_c|
        photo_loss_left = self.photo_loss(I_l_warp, I_c)
        photo_loss_right = self.photo_loss(I_r_warp, I_c)
        
        wpm_loss = (O_l * photo_loss_left).mean() + (O_r * photo_loss_right).mean()
        loss_dict['wpm'] = wpm_loss
        
        # 2. Reconstruction Loss (ℓ_rec)
        # I_rec = O_l ⊙ I_l→c + O_r ⊙ I_r→c
        # ℓ_rec = ||I_rec - I_c||₁
        I_rec = O_l * I_l_warp + O_r * I_r_warp
        rec_loss = F.l1_loss(I_rec, I_c)
        loss_dict['rec'] = rec_loss
        
        # 3. SSIM Loss (ℓ_SSIM)
        # ℓ_SSIM = 1 - (SSIM(I_l→c, I_c) + SSIM(I_r→c, I_c)) / 2
        ssim_loss_left = self.ssim_loss(I_l_warp, I_c)
        ssim_loss_right = self.ssim_loss(I_r_warp, I_c)
        ssim_loss = (ssim_loss_left + ssim_loss_right) / 2
        loss_dict['ssim'] = ssim_loss
        
        # 4. Disparity Smoothness Loss (ℓ_smd)
        # ℓ_smd = (1/XY) Σ |∇d̃| ⊙ exp(-η|∇I_c|)
        smd_loss = self.smoothness_loss(d̃, I_c)
        
        # 也对 coarse disparity 计算平滑性损失
        smd_loss_coarse = self.smoothness_loss(d̃_coarse, I_c)
        smd_loss = smd_loss + smd_loss_coarse
        loss_dict['smd'] = smd_loss
        
        # 5. Occlusion Smoothness Loss (ℓ_smo)
        # ℓ_smo = (1/XY) Σ |∇O_l| ⊙ exp(-η|∇I_c|)
        smo_loss = self.smoothness_loss(O_l, I_c)
        loss_dict['smo'] = smo_loss
        
        # 总损失
        # ℓ_full = ℓ_wpm + ℓ_rec + α₁ℓ_SSIM + α₂ℓ_smd + α₃ℓ_smo
        total_loss = (
            self.lambda_wpm * wpm_loss +
            self.lambda_rec * rec_loss +
            self.lambda_ssim * ssim_loss +
            self.lambda_smd * smd_loss +
            self.lambda_smo * smo_loss
        )
        
        loss_dict['total'] = total_loss
        
        return loss_dict


def build_loss(config: dict = None) -> UnsupervisedLoss:
    """构建损失函数"""
    return UnsupervisedLoss(config)


if __name__ == '__main__':
    """测试损失函数"""
    print("测试损失函数模块...")
    
    # 创建测试数据
    batch_size = 2
    H, W = 256, 256
    
    views = [
        torch.randn(batch_size, 3, H, W),  # I_left
        torch.randn(batch_size, 3, H, W),  # I_center
        torch.randn(batch_size, 3, H, W)   # I_right
    ]
    
    disp_outputs = {
        'disparity': torch.rand(batch_size, 1, H, W) * 10,
        'disparity_coarse': torch.rand(batch_size, 1, H, W) * 10
    }
    
    occ_outputs = {
        'occlusion_left': torch.rand(batch_size, 1, H, W),
        'occlusion_right': torch.rand(batch_size, 1, H, W),
        'occlusion_maps': torch.rand(batch_size, 2, H, W)
    }
    
    warped_views = {
        'left': torch.randn(batch_size, 3, H, W),
        'right': torch.randn(batch_size, 3, H, W)
    }
    
    # 创建损失函数
    loss_fn = build_loss()
    
    # 计算损失
    loss_dict = loss_fn(disp_outputs, occ_outputs, views, warped_views)
    
    print("\n损失计算结果:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    print("\n✓ 损失函数测试通过!")

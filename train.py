#!/usr/bin/env python3
"""
训练脚本

无监督光场深度估计训练
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.dispnet import build_dispnet
from src.occnet import build_occnet
from src.loss import build_loss
from src.data import HCIDataset, warp_view


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: Path,
                   filename: str = 'checkpoint.pth.tar'):
    """保存检查点"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = checkpoint_dir / 'model_best.pth.tar'
        torch.save(state, best_filepath)


def train_epoch(model_disp: nn.Module, model_occ: nn.Module,
                criterion: nn.Module, optimizer: optim.Optimizer,
                dataloader: DataLoader, device: torch.device,
                epoch: int, writer: SummaryWriter, args) -> dict:
    """训练一个 epoch"""
    
    model_disp.train()
    model_occ.train()
    
    total_loss = 0
    loss_dict = {
        'total': 0, 'wpm': 0, 'rec': 0, 'ssim': 0, 'smd': 0, 'smo': 0
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        views = batch['views'].to(device)  # [B, N_views, C, H, W]
        
        # 选择三视图：左、中、右
        # 假设视图按顺序排列
        batch_size = views.size(0)
        n_views = views.size(1)
        
        # 选择中心视图和左右视图
        center_idx = n_views // 2
        left_idx = center_idx - 1
        right_idx = center_idx + 1
        
        I_left = views[:, left_idx]  # [B, C, H, W]
        I_center = views[:, center_idx]
        I_right = views[:, right_idx]
        
        views_list = [I_left, I_center, I_right]
        
        # 前向传播
        optimizer.zero_grad()
        
        # DispNet
        disp_outputs = model_disp(views_list)
        disparity = disp_outputs['disparity']
        
        # Warp 视图
        I_left_warp = warp_view(I_left, disparity, direction='horizontal')
        I_right_warp = warp_view(I_right, disparity, direction='horizontal')
        
        warped_views = {
            'left': I_left_warp,
            'right': I_right_warp
        }
        
        # OccNet
        occ_outputs = model_occ(I_left_warp, I_right_warp, disparity)
        
        # 计算损失
        loss_data = criterion(disp_outputs, occ_outputs, views_list, warped_views)
        
        # 反向传播
        loss = loss_data['total']
        loss.backward()
        
        # 梯度裁剪
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model_disp.parameters()) + list(model_occ.parameters()),
                args.grad_clip
            )
        
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        for key in loss_dict:
            if key in loss_data:
                loss_dict[key] += loss_data[key].item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'wpm': f'{loss_data["wpm"].item():.4f}',
            'ssim': f'{loss_data["ssim"].item():.4f}'
        })
        
        # TensorBoard
        if batch_idx % args.log_interval == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/total_loss', loss.item(), step)
            writer.add_scalar('train/wpm_loss', loss_data['wpm'].item(), step)
            writer.add_scalar('train/rec_loss', loss_data['rec'].item(), step)
            writer.add_scalar('train/ssim_loss', loss_data['ssim'].item(), step)
            writer.add_scalar('train/smd_loss', loss_data['smd'].item(), step)
            writer.add_scalar('train/smo_loss', loss_data['smo'].item(), step)
    
    # 计算平均损失
    n_batches = len(dataloader)
    avg_loss = {key: value / n_batches for key, value in loss_dict.items()}
    
    return avg_loss


@torch.no_grad()
def validate(model_disp: nn.Module, model_occ: nn.Module,
             criterion: nn.Module, dataloader: DataLoader,
             device: torch.device, epoch: int,
             writer: SummaryWriter) -> dict:
    """验证"""
    
    model_disp.eval()
    model_occ.eval()
    
    total_loss = 0
    loss_dict = {
        'total': 0, 'wpm': 0, 'rec': 0, 'ssim': 0, 'smd': 0, 'smo': 0
    }
    
    pbar = tqdm(dataloader, desc=f'Validation')
    
    for batch in pbar:
        views = batch['views'].to(device)
        batch_size = views.size(0)
        n_views = views.size(1)
        
        center_idx = n_views // 2
        left_idx = center_idx - 1
        right_idx = center_idx + 1
        
        I_left = views[:, left_idx]
        I_center = views[:, center_idx]
        I_right = views[:, right_idx]
        
        views_list = [I_left, I_center, I_right]
        
        # 前向传播
        disp_outputs = model_disp(views_list)
        disparity = disp_outputs['disparity']
        
        I_left_warp = warp_view(I_left, disparity, direction='horizontal')
        I_right_warp = warp_view(I_right, disparity, direction='horizontal')
        
        warped_views = {
            'left': I_left_warp,
            'right': I_right_warp
        }
        
        occ_outputs = model_occ(I_left_warp, I_right_warp, disparity)
        
        # 计算损失
        loss_data = criterion(disp_outputs, occ_outputs, views_list, warped_views)
        
        # 记录损失
        total_loss += loss_data['total'].item()
        for key in loss_dict:
            if key in loss_data:
                loss_dict[key] += loss_data[key].item()
    
    # 计算平均损失
    n_batches = len(dataloader)
    avg_loss = {key: value / n_batches for key, value in loss_dict.items()}
    
    # TensorBoard
    step = epoch * len(dataloader)
    writer.add_scalar('val/total_loss', avg_loss['total'], step)
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='DispNet-OccNet 训练')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录（覆盖配置文件）')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='梯度裁剪阈值')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='日志记录间隔')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.data_dir:
        config['dataset']['data_dir'] = args.data_dir
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / config['experiment']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    
    # TensorBoard
    writer = SummaryWriter(str(log_dir))
    
    # 构建模型
    print('构建模型...')
    model_disp = build_dispnet(config['dispnet']).to(device)
    model_occ = build_occnet(config['occnet']).to(device)
    
    # 计算参数量
    disp_params = sum(p.numel() for p in model_disp.parameters())
    occ_params = sum(p.numel() for p in model_occ.parameters())
    print(f'DispNet 参数量：{disp_params:,} ({disp_params/1e6:.3f}M)')
    print(f'OccNet 参数量：{occ_params:,} ({occ_params/1e6:.3f}M)')
    
    # 优化器
    params = list(model_disp.parameters()) + list(model_occ.parameters())
    optimizer = optim.Adam(params, lr=config['optimizer']['lr'],
                          betas=tuple(config['optimizer']['betas']),
                          weight_decay=config['optimizer']['weight_decay'])
    
    # 学习率调度
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['scheduler']['step_size'],
        gamma=config['scheduler']['gamma']
    )
    
    # 损失函数
    criterion = build_loss(config['loss'])
    
    # 数据加载
    print('加载数据...')
    train_dataset = HCIDataset(
        data_dir=config['dataset']['data_dir'],
        split='train',
        augment=config['dataset']['augment'],
        crop_size=tuple(config['dataset']['crop_size'])
    )
    
    val_dataset = HCIDataset(
        data_dir=config['dataset']['data_dir'],
        split='val',
        augment=False,
        crop_size=tuple(config['dataset']['crop_size'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=True,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True
    )
    
    print(f'训练集大小：{len(train_dataset)}')
    print(f'验证集大小：{len(val_dataset)}')
    
    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model_disp.load_state_dict(checkpoint['dispnet'])
        model_occ.load_state_dict(checkpoint['occnet'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f'从 epoch {start_epoch} 恢复训练')
    
    # 训练循环
    print('开始训练...')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f'\nEpoch {epoch + 1}/{config["training"]["epochs"]}')
        
        # 训练
        train_loss = train_epoch(
            model_disp, model_occ, criterion, optimizer,
            train_loader, device, epoch, writer, args
        )
        
        # 学习率更新
        scheduler.step()
        
        # 验证
        if (epoch + 1) % config['training']['val_interval'] == 0:
            val_loss = validate(
                model_disp, model_occ, criterion,
                val_loader, device, epoch, writer
            )
            
            print(f'验证损失：{val_loss["total"]:.6f}')
        
        # 保存检查点
        is_best = False
        if val_loss['total'] < best_loss:
            best_loss = val_loss['total']
            is_best = True
        
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'dispnet': model_disp.state_dict(),
                'occnet': model_occ.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
                'config': config
            }, is_best, checkpoint_dir)
    
    print('\n训练完成!')
    writer.close()


if __name__ == '__main__':
    main()

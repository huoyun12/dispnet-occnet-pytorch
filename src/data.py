"""
光场数据加载器

支持的数据集：
- HCI Dataset (密集光场)
- DLF Dataset (密集光场)
- SLF Dataset (稀疏光场)
- Stanford Lytro Archive (真实世界光场)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict
import os


class LightFieldDataset(Dataset):
    """
    光场深度估计数据集
    
    加载多视角光场图像，支持数据增强和预处理
    """
    
    def __init__(self, data_dir: str, split: str = 'train',
                 max_disparity: int = 12, view_range: Tuple[int, int] = (0, 7),
                 crop_size: Tuple[int, int] = (256, 256),
                 augment: bool = True, transform=None):
        """
        Args:
            data_dir: 数据根目录
            split: 数据集划分 ('train', 'val', 'test')
            max_disparity: 最大视差值
            view_range: 使用的视角范围
            crop_size: 裁剪尺寸 (H, W)
            augment: 是否使用数据增强
            transform: 数据变换
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_disparity = max_disparity
        self.view_range = view_range
        self.crop_size = crop_size
        self.augment = augment
        self.transform = transform
        
        # 加载样本列表
        self.samples = self._load_sample_list()
    
    def _load_sample_list(self) -> List[str]:
        """加载样本列表"""
        samples_file = self.data_dir / f'{self.split}.txt'
        
        if samples_file.exists():
            with open(samples_file, 'r') as f:
                samples = [line.strip() for line in f.readlines()]
        else:
            # 自动扫描目录
            samples = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_name = self.samples[idx]
        sample_dir = self.data_dir / sample_name
        
        # 加载多视角图像
        views = []
        
        # 尝试两种格式
        # 格式 1: imgVVUU.png (HCI 原始格式)
        # 格式 2: input_CamXXX.png (HCI 新格式)
        
        # 检查格式
        test_path1 = sample_dir / 'img0000.png'
        test_path2 = sample_dir / 'input_Cam000.png'
        
        use_old_format = test_path1.exists()
        
        if use_old_format:
            # 旧格式：imgVVUU.png
            for v in range(self.view_range[0], self.view_range[1]):
                for u in range(self.view_range[0], self.view_range[1]):
                    view_path = sample_dir / f'img{v:02d}{u:02d}.png'
                    if view_path.exists():
                        view = Image.open(view_path).convert('RGB')
                        view = np.array(view).astype(np.float32) / 255.0
                        views.append(view)
        else:
            # 新格式：input_CamXXX.png
            # 假设 Cam000 是中心视图，按顺序加载
            cam_indices = list(range(0, 81))  # 最多 81 个视图 (9x9)
            
            for cam_idx in cam_indices:
                view_path = sample_dir / f'input_Cam{cam_idx:03d}.png'
                if view_path.exists():
                    view = Image.open(view_path).convert('RGB')
                    view = np.array(view).astype(np.float32) / 255.0
                    views.append(view)
                
                # 如果已经加载了足够的视图，停止
                if len(views) >= self.view_range[1] * self.view_range[1]:
                    break
        
        # 加载深度图（如果有，仅用于评估）
        depth_path = sample_dir / 'depth0.png'
        if depth_path.exists():
            depth = np.array(Image.open(depth_path)).astype(np.float32)
        else:
            depth = None
        
        # 数据增强
        if self.augment:
            views, depth = self._augment(views, depth)
        
        # 随机裁剪
        if self.crop_size is not None:
            views, depth = self._random_crop(views, depth)
        
        # 转换为 tensor
        views_tensor = torch.stack([
            torch.from_numpy(v).permute(2, 0, 1).float() for v in views
        ])
        
        sample = {
            'views': views_tensor,
            'sample_name': sample_name,
        }
        
        if depth is not None:
            sample['depth'] = torch.from_numpy(depth)
        
        return sample
    
    def _augment(self, views: List[np.ndarray], 
                 depth: np.ndarray = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """数据增强"""
        # 随机水平翻转
        if np.random.random() > 0.5:
            views = [np.fliplr(view) for view in views]
            if depth is not None:
                depth = np.fliplr(depth)
        
        # 随机垂直翻转
        if np.random.random() > 0.5:
            views = [np.flipud(view) for view in views]
            if depth is not None:
                depth = np.flipud(depth)
        
        # 随机亮度和对比度调整
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-0.1, 0.1)
            views = [np.clip(view * alpha + beta, 0, 1) for view in views]
        
        return views, depth
    
    def _random_crop(self, views: List[np.ndarray], 
                     depth: np.ndarray = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """随机裁剪"""
        h, w = views[0].shape[:2]
        crop_h, crop_w = self.crop_size
        
        if h > crop_h or w > crop_w:
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
            
            views = [view[top:top+crop_h, left:left+crop_w] for view in views]
            
            if depth is not None:
                depth = depth[top:top+crop_h, left:left+crop_w]
        
        return views, depth


class HCIDataset(LightFieldDataset):
    """
    HCI 光场数据集
    
    密集光场，7x7 视角，视差范围 [-4, 4]
    """
    
    def __init__(self, data_dir: str, **kwargs):
        # HCI 默认参数
        default_kwargs = {
            'max_disparity': 12,
            'view_range': (0, 7),
            'crop_size': (256, 256)
        }
        default_kwargs.update(kwargs)
        
        super().__init__(data_dir, **default_kwargs)
        
        # HCI 数据集参数
        self.baseline = 0.0375  # 基线距离 (米)
        self.focal_length = 0.009  # 焦距 (米)
    
    def depth_to_disparity(self, depth: np.ndarray) -> np.ndarray:
        """深度转视差"""
        disparity = self.baseline * self.focal_length / (depth + 1e-8)
        return disparity


def warp_view(view: torch.Tensor, disparity: torch.Tensor, 
              direction: str = 'horizontal') -> torch.Tensor:
    """
    根据视差 warp 视图
    
    Args:
        view: 输入视图 [B, C, H, W]
        disparity: 视差图 [B, 1, H, W]
        direction: warp 方向 ('horizontal' 或 'vertical')
    
    Returns:
        warped_view: Warp 后的视图 [B, C, H, W]
    """
    B, C, H, W = view.shape
    
    # 生成采样网格
    y_coord = torch.linspace(-1, 1, H).to(view.device)
    x_coord = torch.linspace(-1, 1, W).to(view.device)
    grid_y, grid_x = torch.meshgrid(y_coord, x_coord, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    # 根据视差偏移
    if direction == 'horizontal':
        grid[:, :, :, 0] = grid[:, :, :, 0] - 2 * disparity.squeeze(1) / W
    elif direction == 'vertical':
        grid[:, :, :, 1] = grid[:, :, :, 1] - 2 * disparity.squeeze(1) / H
    
    # Grid sample
    warped = torch.nn.functional.grid_sample(
        view, grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    return warped


def create_dataloader(data_dir: str, batch_size: int = 4,
                     num_workers: int = 4, **kwargs) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: batch size
        num_workers: worker 数量
        **kwargs: 传递给数据集的参数
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LightFieldDataset(
        data_dir=data_dir,
        transform=transform,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


if __name__ == '__main__':
    """测试数据加载器"""
    print("测试数据加载器...")
    
    # 创建测试数据
    test_dir = Path('test_data')
    test_dir.mkdir(exist_ok=True)
    
    # 创建模拟样本
    sample_dir = test_dir / 'test_sample'
    sample_dir.mkdir(exist_ok=True)
    
    # 创建模拟视图
    for v in range(7):
        for u in range(7):
            view = np.random.rand(256, 256, 3).astype(np.float32)
            view_img = Image.fromarray((view * 255).astype(np.uint8))
            view_path = sample_dir / f'img{v:02d}{u:02d}.png'
            view_img.save(view_path)
    
    # 创建深度图
    depth = np.random.rand(256, 256).astype(np.float32)
    depth_img = Image.fromarray((depth * 1000).astype(np.uint16))
    depth_img.save(sample_dir / 'depth0.png')
    
    # 创建 train.txt
    with open(test_dir / 'train.txt', 'w') as f:
        f.write('test_sample\n')
    
    # 测试数据集
    dataset = HCIDataset(str(test_dir), split='train', augment=False)
    
    print(f"数据集大小：{len(dataset)}")
    
    # 测试数据加载
    sample = dataset[0]
    print(f"视图数量：{sample['views'].shape[0]}")
    print(f"视图形状：{sample['views'].shape[1:]}")
    
    if 'depth' in sample:
        print(f"深度图形状：{sample['depth'].shape}")
    
    # 清理
    import shutil
    shutil.rmtree(test_dir)
    
    print("\n✓ 数据加载器测试通过!")

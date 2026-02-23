#!/usr/bin/env python3
"""
环境配置脚本 - DispNet-OccNet 论文复现

用于设置论文复现所需的 Python 环境和依赖。
"""

import subprocess
import sys
from pathlib import Path


def create_requirements_file(output_path='requirements.txt'):
    """创建 requirements.txt 文件"""
    requirements = """# PyTorch and torchvision
torch>=1.9.0
torchvision>=0.10.0

# Scientific computing
numpy>=1.19.0
scipy>=1.7.0
scikit-image>=0.18.0

# Image processing
opencv-python>=4.5.0
Pillow>=8.0.0
imageio>=2.9.0

# Visualization
matplotlib>=3.4.0
tensorboard>=2.6.0
tqdm>=4.62.0

# Data handling
pyyaml>=5.4.0
h5py>=3.4.0

# PDF processing (for paper analysis)
PyMuPDF>=1.19.0

# Light field specific
# lfr  # 如果需要读取 LFR 格式光场
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print(f"✓ requirements.txt 已创建：{output_path}")


def verify_installation():
    """验证安装"""
    print("\n验证安装...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
        
        import torchvision
        print(f"✓ Torchvision {torchvision.__version__}")
        
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        import yaml
        print(f"✓ PyYAML {yaml.__version__}")
        
        print("\n✓ 所有包验证成功！")
        return True
        
    except ImportError as e:
        print(f"✗ 验证失败：{e}")
        return False


def main():
    print("="*60)
    print("DispNet-OccNet 论文复现 - 环境配置")
    print("="*60)
    print("\n论文：Unsupervised Light Field Depth Estimation via")
    print("      Multi-view Feature Matching with Occlusion Prediction")
    print("作者：Shansi Zhang, Nan Meng, Edmund Y. Lam")
    print("发表：IEEE Transactions on Image Processing")
    print("="*60)
    
    # 创建 requirements.txt
    create_requirements_file()
    
    print("\n已创建 requirements.txt 文件")
    print("\n安装依赖:")
    print("  pip install -r requirements.txt")
    
    # 验证安装
    choice = input("\n是否验证当前安装？(y/n): ")
    if choice.lower() == 'y':
        verify_installation()
    
    print("\n" + "="*60)
    print("下一步:")
    print("1. 安装依赖：pip install -r requirements.txt")
    print("2. 下载数据集：参考 references/data_processing.md")
    print("3. 阅读论文总结：references/paper_summary.md")
    print("4. 开始复现：参考 SKILL.md 中的复现步骤")
    print("="*60)


if __name__ == '__main__':
    main()

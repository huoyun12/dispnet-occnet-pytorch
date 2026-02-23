#!/usr/bin/env python3
"""
检查和修复 PyTorch 环境问题
"""

import sys
import os

print("="*60)
print("PyTorch 环境检查")
print("="*60)

# 1. 检查 Python 版本
print(f"\n1. Python 版本：{sys.version}")
print(f"   路径：{sys.executable}")

# 2. 检查系统信息
import platform
print(f"\n2. 系统：{platform.system()} {platform.release()}")
print(f"   架构：{platform.machine()}")

# 3. 检查环境变量
print(f"\n3. 环境变量检查:")
print(f"   PATH 长度：{len(os.environ.get('PATH', ''))}")

# 4. 尝试导入 PyTorch
print(f"\n4. 尝试导入 PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch 版本：{torch.__version__}")
    print(f"   ✓ CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA 版本：{torch.version.cuda}")
        print(f"   ✓ GPU 数量：{torch.cuda.device_count()}")
except Exception as e:
    print(f"   ✗ PyTorch 导入失败：{e}")
    print(f"\n   建议解决方案:")
    print(f"   1. 重新安装 PyTorch:")
    print(f"      pip uninstall torch torchvision")
    print(f"      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print(f"   2. 或者使用 CPU 版本:")
    print(f"      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    print(f"   3. 检查 Visual C++ Redistributable 是否安装")

# 5. 检查其他依赖
print(f"\n5. 检查其他依赖:")
deps = {
    'numpy': '数值计算',
    'PIL': '图像处理',
    'yaml': '配置文件',
    'tqdm': '进度条',
    'tensorboard': '可视化'
}

for module, desc in deps.items():
    try:
        __import__(module)
        print(f"   ✓ {module}: {desc}")
    except ImportError:
        print(f"   ✗ {module}: 未安装 ({desc})")

print("\n" + "="*60)

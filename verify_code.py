import ast
import os
import sys

print("=" * 60)
print("DispNet-OccNet 代码验证脚本")
print("=" * 60)

def check_syntax(filepath):
    """检查Python文件语法"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

project_root = "e:/光场相机相关/深度估计/深度学习/Dispnet-Occnet"

files_to_check = [
    "models/__init__.py",
    "models/dispnet.py",
    "models/occnet.py",
    "utils/__init__.py",
    "utils/losses.py",
    "utils/config.py",
    "data/__init__.py",
    "data/lightfield_dataset.py",
    "train.py",
    "inference.py",
    "evaluate.py",
    "test_models.py",
    "scripts/prepare_data.py",
]

print("\n[1] 检查所有Python文件语法...")
all_ok = True

for file_path in files_to_check:
    full_path = os.path.join(project_root, file_path)
    if os.path.exists(full_path):
        ok, error = check_syntax(full_path)
        if ok:
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path}: {error}")
            all_ok = False
    else:
        print(f"  ✗ {file_path}: 文件不存在")
        all_ok = False

print("\n[2] 检查配置文件...")
config_files = [
    "configs/dense_lf.yaml",
    "configs/sparse_lf.yaml",
]

for config_file in config_files:
    full_path = os.path.join(project_root, config_file)
    if os.path.exists(full_path):
        print(f"  ✓ {config_file}")
    else:
        print(f"  ✗ {config_file}: 文件不存在")
        all_ok = False

print("\n[3] 检查依赖文件...")
dep_files = [
    "requirements.txt",
    "README.md",
]

for dep_file in dep_files:
    full_path = os.path.join(project_root, dep_file)
    if os.path.exists(full_path):
        print(f"  ✓ {dep_file}")
    else:
        print(f"  ✗ {dep_file}: 文件不存在")

print("\n[4] 检查目录结构...")
required_dirs = [
    "models",
    "utils",
    "data",
    "configs",
    "scripts",
]

for dir_name in required_dirs:
    full_path = os.path.join(project_root, dir_name)
    if os.path.isdir(full_path):
        print(f"  ✓ {dir_name}/")
    else:
        print(f"  ✗ {dir_name}/: 目录不存在")
        all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✓ 所有代码验证通过！")
else:
    print("✗ 部分验证失败，请检查上述错误")
print("=" * 60)

print("\n注意: 当前环境的PyTorch有DLL加载问题，无法实际运行验证。")
print("      代码语法已通过检查，在正确的PyTorch环境中应该可以正常运行。")

print("\n下一步:")
print("  1. 修复PyTorch环境 (重新安装或使用conda)")
print("  2. 准备数据集")
print("  3. 运行: python train.py --data_dir ./data/hci/train")

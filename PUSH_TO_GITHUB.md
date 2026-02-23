# 推送到 GitHub 指南

## 问题说明

GitHub MCP 调用失败，原因是 **Personal Access Token 权限不足**。

## 解决方案

### 方案 1：使用 Git 命令行推送（推荐）

#### 1. 配置 Git（如果还没配置）

```powershell
cd "e:\光场相机相关\深度估计\深度学习\Dispnet-Occnet"

# 配置用户信息
git config user.name "huoyun12"
git config user.email "2940194378@qq.com"
```

#### 2. 添加远程仓库

```powershell
# 添加远程仓库
git remote add origin https://github.com/huoyun12/dispnet-occnet-pytorch.git

# 验证
git remote -v
```

#### 3. 添加文件并提交

```powershell
# 添加所有文件
git add .

# 或者只添加重要文件
git add README.md
git add .gitignore
git add src/
git add test_simple.py
git add log.md
git add requirements.txt

# 提交
git commit -m "Initial commit: Add DispNet and OccNet implementation

- Add DispNet (disparity estimation network)
- Add OccNet (occlusion prediction network)  
- Add project structure and README
- Add tests (both networks pass)
- Add progress log

Network parameters:
- DispNet: ~1.8M (matches paper)
- OccNet: ~0.11M (matches paper)

Status: Core networks implemented and tested ✓"
```

#### 4. 推送到 GitHub

```powershell
# 如果默认分支是 main
git push -u origin main

# 如果默认分支是 master
git push -u origin master
```

#### 5. 输入 GitHub 凭证

推送时会提示输入用户名和密码：
- **Username**: huoyun12
- **Password**: 使用 Personal Access Token（不是 GitHub 密码）

### 方案 2：使用 GitHub Desktop

1. 下载并安装 [GitHub Desktop](https://desktop.github.com/)
2. 登录 GitHub 账号
3. 添加本地仓库：File → Add Local Repository
4. 选择项目目录：`e:\光场相机相关\深度估计\深度学习\Dispnet-Occnet`
5. 推送到 GitHub

### 方案 3：使用 VS Code

1. 在 VS Code 中打开项目
2. 点击左侧 Source Control 图标
3. 点击 "Publish to GitHub"
4. 选择 Public 或 Private
5. 完成推送

## Personal Access Token 配置

如果还没有 Personal Access Token：

1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 填写说明（如：dispnet-occnet-project）
4. 选择权限：
   - ✅ `repo` (Full control of private repositories)
   - ✅ `workflow` (Update GitHub Action workflows)
5. 点击 "Generate token"
6. **重要**：复制并保存 token（只显示一次）

## 验证推送

推送成功后，访问：
https://github.com/huoyun12/dispnet-occnet-pytorch

应该能看到：
- README.md
- src/dispnet.py
- src/occnet.py
- src/__init__.py
- .gitignore
- log.md
- test_simple.py

## 常见问题

### Q: 提示 "remote origin already exists"

```powershell
# 删除现有 remote
git remote remove origin

# 重新添加
git remote add origin https://github.com/huoyun12/dispnet-occnet-pytorch.git
```

### Q: 推送失败，提示权限错误

- 检查 Personal Access Token 是否有 `repo` 权限
- 确认用户名和 token 正确
- 尝试重新生成 token

### Q: 分支名称不对

```powershell
# 查看当前分支
git branch

# 重命名分支
git branch -M main

# 推送
git push -u origin main
```

## 推送后的下一步

推送成功后：
1. 在 GitHub 上查看代码
2. 继续实现剩余功能（损失函数、数据加载器等）
3. 定期提交和推送进度

---

**创建时间**: 2026-02-23
**最后更新**: 2026-02-23

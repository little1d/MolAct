# GitHub Repository Setup Guide

## 推送到 GitHub 的步骤

### 1. 在 GitHub 上创建新仓库

1. 访问 https://github.com/new
2. 仓库名称: `MolAct`
3. 描述: "Molecular Action Agents for Chemistry Tasks"
4. 选择 Public 或 Private
5. **不要**初始化 README、.gitignore 或 LICENSE（我们已经有了）
6. 点击 "Create repository"

### 2. 添加远程仓库并推送

```bash
cd /mnt/shared-storage-user/yangzhuo/main/projects/MolAct

# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/MolAct.git

# 或者使用 SSH
# git remote add origin git@github.com:YOUR_USERNAME/MolAct.git

# 提交所有文件
git add -A
git commit -m "Initial commit: inference and evaluation code for MolAct"

# 推送到 GitHub
git push -u origin main
```

### 3. 验证

访问 `https://github.com/YOUR_USERNAME/MolAct` 确认所有文件都已上传。

## 仓库内容说明

### 包含的内容
- ✅ `agentfly/` - 核心框架（已排除 rewards 目录）
- ✅ `scripts/` - 推理和评估脚本
- ✅ `ChemCoTBench/` - 评估框架
- ✅ `oracle/` - Oracle 数据文件
- ✅ `README.md` - 项目说明
- ✅ `requirements.txt` - 依赖列表
- ✅ `.gitignore` - Git 忽略规则

### 不包含的内容
- ❌ `verl/` - 训练框架（submodule，不包含）
- ❌ `agentfly/rewards/` - 训练 reward 函数
- ❌ `data/` - 训练数据（已在 Hugging Face 上）
- ❌ `outputs/` - 输出结果
- ❌ `checkpoints/` - 模型检查点

## 后续更新

如果需要更新代码：

```bash
cd /mnt/shared-storage-user/yangzhuo/main/projects/MolAct

# 修改文件后
git add -A
git commit -m "Update: description of changes"
git push origin main
```

## 注意事项

1. **不要提交敏感信息**：确保 `.gitignore` 正确配置
2. **检查文件大小**：GitHub 有文件大小限制（100MB），大文件应使用 Git LFS
3. **Oracle 文件**：oracle 目录中的 .pkl 文件较大，如果超过限制，可以考虑：
   - 使用 Git LFS
   - 或者只上传到 Hugging Face，在 README 中提供下载链接


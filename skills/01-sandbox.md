# Skill: Sandbox (沙盒隔离)

**目标**：在修改任何代码之前，创建一个完全隔离的沙盒环境。

---

## 为什么需要沙盒

优化深度学习训练代码有高风险：
- 可能破坏原有代码结构
- 可能引入难以调试的 bug
- 可能影响模型精度

**沙盒确保所有实验都在隔离环境中进行，原项目零风险。**

---

## 执行步骤

### Step 1: 确定项目路径

```bash
# 确认要优化的项目路径
PROJECT_ROOT="/path/to/training/project"
```

### Step 2: 创建沙盒目录

```bash
SANDBOX_ROOT="/tmp/trainflashagent_sandboxes/<project_name>"
mkdir -p "$SANDBOX_ROOT"
```

### Step 3: 克隆项目

```bash
# 使用 rsync 或 cp 进行完整克隆
rsync -av --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
      "$PROJECT_ROOT/" "$SANDBOX_ROOT/"
```

### Step 4: 验证克隆

```bash
# 确认关键文件存在
ls "$SANDBOX_ROOT/train.py"
ls "$SANDBOX_ROOT/requirements.txt"
```

---

## 检查清单

在执行任何优化之前，确认：

- [ ] 沙盒目录已创建
- [ ] 关键训练脚本已复制
- [ ] 配置文件已复制
- [ ] 原项目路径未被修改

---

## 回滚方案

如果沙盒中的修改失败：

```bash
# 删除沙盒，重新克隆
rm -rf "$SANDBOX_ROOT"
rsync -av "$PROJECT_ROOT/" "$SANDBOX_ROOT/"
```

---

## 输出格式

成功后，记录：

```
[Sandbox Created]
- Project: /path/to/training/project
- Sandbox: /tmp/trainflashagent_sandboxes/<project_name>
- Timestamp: 2026-04-07T10:30:00
```

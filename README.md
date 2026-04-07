# TrainFlashAgent ⚡

**一套让 AI 像顶级性能工程师一样优化深度学习训练的 Skills。**

这不是一个 Python 库，也不是一个 MCP Server。这是一套 **Markdown 规范**，可以被任何 AI 编辑器（Cursor、GitHub Copilot、Claude Code）直接使用。

---

## 🚀 为什么是 Markdown Skills？

| 传统方案 | TrainFlashAgent |
|---------|-----------------|
| 需要安装 Python 库 | **零依赖，复制即用** |
| 需要配置 MCP Server | **任何 AI 编辑器都支持** |
| 预设流程限制 | **LLM 自主决策** |
| 特定平台绑定 | **跨平台、可传播** |

**这就是为什么像 [airbnb/airbnb-cursor-rules](https://github.com/airbnb/airbnb-cursor-rules) 这样的项目能获得 5k+ Star。**

---

## 📁 项目结构

```
TrainFlashAgent/
├── skills/
│   ├── 01-sandbox.md       # 沙盒隔离技能
│   ├── 02-diagnose.md      # 宏观诊断技能
│   ├── 03-optimize.md      # 工程优化技能
│   └── 04-verify.md        # 验证与回写技能
├── .cursorrules            # Cursor 专用配置
├── .github/copilot-instructions.md  # Copilot 专用配置
└── README.md
```

---

## 🛠️ 快速开始

### 方式 1: Cursor (推荐)

1. 复制 `.cursorrules` 到你的项目根目录：
```bash
cp TrainFlashAgent/.cursorrules /path/to/your/project/
```

2. 对 Cursor 说：
> "帮我优化这个项目的训练速度"

Cursor 会自动读取 `.cursorrules`，按照 Skills 流程执行。

---

### 方式 2: GitHub Copilot

1. 复制 `.github/copilot-instructions.md` 到你的 `.github/` 目录：
```bash
mkdir -p .github
cp TrainFlashAgent/.github/copilot-instructions.md .github/
```

2. 在 Copilot Chat 中询问：
> "如何优化我的训练循环？"

---

### 方式 3: Claude Code

1. 将 `skills/` 目录作为上下文提供给 Claude：
```bash
claude --context "TrainFlashAgent/skills/"
```

2. 或者直接引用：
> "参考 TrainFlashAgent 的 02-diagnose.md 技能，帮我诊断训练瓶颈"

---

## 📋 Skills 列表

### Skill 01: Sandbox (沙盒隔离)
**目标**：创建隔离环境，确保原项目零风险。

**关键步骤**：
```bash
rsync -av --exclude '__pycache__' /path/to/project/ /tmp/sandbox/
```

---

### Skill 02: Diagnose (宏观诊断)
**目标**：用手动计时器识别瓶颈，**不要直接调用 Profiler**。

**关键代码**：
```python
import time
start = time.perf_counter()
# ... your code ...
print(f"[TRAINOPT_TIMER] Label: {time.perf_counter() - start:.4f}s")
```

**分析重点**：
- DataLoader 时间占比是否 > 40%？
- 是否存在长尾（max > 2x mean）？

---

### Skill 03: Optimize (工程优化)
**目标**：根据诊断结果应用优化。

**优先级**：
1. **DataLoader 优化** (num_workers, prefetch_factor) - 2-5x 提升
2. **资源分配** (batch_size, mixed precision) - 20-50% 提升
3. **代码模式** (向量化) - 10-30% 提升
4. **算子优化** - **最后手段**

---

### Skill 04: Verify (验证与回写)
**目标**：验证效果并安全回写。

**验收标准**：
- 吞吐量提升 > 5%
- Loss 误差 < 0.001
- **用户确认后才能回写**

---

## 🎯 核心方法论

```
┌─────────────────────────────────────────────────────────┐
│  自顶向下 (Top-Down) 诊断                                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Phase 1: 手动计时器                                     │
│   ↓ 识别宏观瓶颈 (DataLoad/Forward/Backward)              │
│                                                          │
│   Phase 2: 工程优化                                       │
│   ↓ 解决 IO、长尾、资源分配                                │
│                                                          │
│   Phase 3: 仅当 Phase 2 收益趋近于零时，才调用 Profiler   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 💡 典型使用场景

### 场景 1: 训练速度慢，不知道瓶颈在哪

```
用户：帮我优化训练速度

AI (按照 Skills):
1. 创建沙盒
2. 插入手动计时器
3. 运行 100 steps 收集日志
4. 分析发现 DataLoader 占 60% 时间，且有长尾
5. 建议：增加 num_workers=8, prefetch_factor=4
6. 验证：吞吐量从 120 → 180 samples/sec (+50%)
7. 等待用户确认后回写
```

### 场景 2: GPU 利用率低

```
用户：GPU 利用率只有 30%，怎么办？

AI (按照 Skills):
1. 诊断：Forward/Backward 时间太短
2. 建议：增加 batch_size 或启用混合精度
3. 验证：GPU 利用率 → 75%
```

---

## 📚 与其他项目对比

| 项目 | 类型 | Star | 特点 |
|------|------|------|------|
| airbnb/airbnb-cursor-rules | Markdown Rules | 5k+ | React 开发规范 |
| paul-gauthier/aider | CLI 工具 | 15k+ | AI 编程助手 |
| TrainFlashAgent | Markdown Skills | - | 训练性能优化 |

**TrainFlashAgent 的定位**：像 airbnb-cursor-rules 一样，通过 Markdown 规范指导 AI，而不是作为一个独立的工具。

---

## 🤝 贡献

欢迎提交新的 Skills 或改进现有内容！

---

## 📄 License

MIT

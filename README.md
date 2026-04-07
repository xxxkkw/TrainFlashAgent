# TrainFlashAgent

<div align="center">

**让 AI 像资深性能工程师一样优化深度学习训练**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cursor](https://img.shields.io/badge/Cursor-支持-brightgreen)](https://cursor.sh)
[![Copilot](https://img.shields.io/badge/Copilot-支持-blue)](https://github.com/features/copilot)

</div>

---

## 📖 简介

TrainFlashAgent 是一套**基于 Markdown 的性能优化方法论**，专为深度学习训练场景设计。

它不是一个需要安装的库，而是一套可以被任何 AI 编辑器（Cursor、GitHub Copilot、Claude Code）直接读取和执行的规范。

**核心思想**：通过自顶向下的诊断流程，让 AI 自动识别训练瓶颈并优化，同时确保模型精度不受影响。

---

## ⚡ 快速开始

### 1️⃣ 复制配置文件

```bash
# Cursor 用户
cp .cursorrules /path/to/your/project/

# GitHub Copilot 用户
mkdir -p .github && cp .github/copilot-instructions.md .github/
```

### 2️⃣ 对 AI 说

> "帮我优化这个项目的训练速度"

就这么简单。AI 会自动按照 Skills 流程执行：创建沙盒 → 诊断瓶颈 → 应用优化 → 验证效果。

---

## 🔥 核心特性

| 特性 | 说明 |
|------|------|
| **🔒 零风险** | 所有实验在沙盒中进行，原项目不受影响 |
| **📊 自顶向下** | 先宏观诊断，再工程优化，最后才考虑算子级调优 |
| **🤖 AI 原生** | 专为 LLM 设计，AI 自主决策优化策略 |
| **🔧 零依赖** | 纯 Markdown 文档，无需安装任何库 |
| **📱 跨平台** | Cursor / Copilot / Claude Code 通用 |

---

## 🎯 解决的问题

深度学习训练优化常见的痛点：

| 问题 | TrainFlashAgent 方案 |
|------|---------------------|
| 不知道瓶颈在哪 | 手动计时器快速定位（DataLoad/Forward/Backward） |
| 盲目调参效果差 | 基于诊断数据的针对性优化 |
| 担心影响模型精度 | 严格的保真度验证（Loss 误差 < 0.001） |
| 不敢修改代码 | 沙盒隔离 + 用户确认机制 |

---

## 📚 文档结构

```
TrainFlashAgent/
├── skills/                    # 核心技能规范
│   ├── 01-sandbox.md         # 创建隔离沙盒
│   ├── 02-diagnose.md        # 诊断瓶颈
│   ├── 03-optimize.md        # 应用优化
│   └── 04-verify.md          # 验证与回写
├── .cursorrules              # Cursor 配置
├── .github/copilot-instructions.md  # Copilot 配置
└── README.md
```

---

## 🛠️ 使用示例

### 场景 1: 训练速度慢

```
你：帮我优化训练速度

AI:
1. [沙盒] 已创建沙盒：/tmp/trainflash_sandbox/
2. [诊断] 插入计时器，运行 100 steps...
3. [分析] DataLoader 占 60% 时间，存在长尾问题
4. [建议] 推荐：num_workers=8, prefetch_factor=4
5. [验证] 吞吐量：120 → 180 samples/sec (+50%)
6. [等待] 请确认是否回写代码
```

### 场景 2: GPU 利用率低

```
你：GPU 利用率只有 30%，怎么办？

AI:
1. [诊断] Forward/Backward 时间过短
2. [建议] 增加 batch_size 或启用混合精度训练
3. [验证] GPU 利用率提升至 75%
```

---

## 📖 方法论

### 自顶向下诊断

```
┌─────────────────────────────────────┐
│  Phase 1: 手动计时器                 │
│  → 识别宏观瓶颈                       │
├─────────────────────────────────────┤
│  Phase 2: 工程优化                   │
│  → DataLoader / 资源分配 / 代码模式   │
├─────────────────────────────────────┤
│  Phase 3: Profiler (最后手段)        │
│  → 仅当 Phase 2 收益趋近于零时调用    │
└─────────────────────────────────────┘
```

### 优化优先级

1. **数据管道** (num_workers, prefetch_factor) - 通常 2-5x 提升
2. **资源分配** (batch_size, mixed precision) - 通常 20-50% 提升
3. **代码模式** (向量化) - 通常 10-30% 提升
4. **算子优化** - 最后考虑，收益有限

---

## 📋 验收标准

每次优化后需要确认：

- ✅ 吞吐量有提升（具体幅度视场景而定）
- ✅ Loss 误差 < 0.001（确保模型精度不受影响）
- ✅ 用户确认后才能回写

---

## 📄 License

MIT License

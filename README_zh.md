# TrainFlashAgent

<div align="center">

**让 AI 像资深性能工程师一样，系统化优化深度学习训练性能**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cursor](https://img.shields.io/badge/Cursor-支持-brightgreen)](https://cursor.sh)
[![Copilot](https://img.shields.io/badge/Copilot-支持-blue)](https://github.com/features/copilot)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-支持-purple)](https://claude.ai/code)

</div>

中文 | [English](README.md)

## 简介
TrainFlashAgent 是一套面向深度学习训练提速的 Skills（Markdown playbooks）。
它定义了一条可执行的优化流程：先用低开销诊断拿到证据，再做工程与训练策略层优化，最后用可复现的指标完成验证与回写治理。

覆盖内容：
- 宏观定位瓶颈（数据/计算/同步/训练环路开销）
- 工程与训练策略优化（effective batch、sampler/bucketing、shape 稳定化、eval/ckpt cadence 等）
- 结果验证与回写治理（指标对比、保真阈值、审批后回写）

定位补充：本项目强调工程与训练策略优化；AMP / compile / TF32 可作为可选手段，但不作为核心卖点或默认路径。

## 设计原则
- 证据优先：先用低开销计时拿到可解释的数据，再决定优化方向。
- 自顶向下：先解决数据管道、训练策略与同步阻塞，再考虑 profiler 与算子级调优。
- 安全默认：所有实验在沙盒完成；回写原项目必须明确审批。
- 可审计迭代：先给出带文件清单的计划；每次只改一个变量；保留前后对比与优化日志。

## 快速开始
1) 将指令文件复制到你要优化的训练项目内：

```bash
# Cursor 用户
cp .cursorrules /path/to/your/project/

# GitHub Copilot 用户
mkdir -p /path/to/your/project/.github
cp .github/copilot-instructions.md /path/to/your/project/.github/

# Claude Code 用户
cp CLAUDE.md /path/to/your/project/
```

2) 对 AI 说：

> “帮我优化这个项目的训练速度，按 TrainFlashAgent 的 skills 流程执行。”

期望 AI 按顺序执行：Sandbox → Diagnose → Optimize → Verify →（可选）Write back。

## 基于 MCP 的诊断助手
TrainFlashAgent 现在内置了一个独立的诊断 MCP 包：`tools/trainflash_mcp`。

它用于在重型 profiler 之前，先做低开销训练诊断，并可组合以下证据：
- 基于 NVML 的 GPU 遥测：核心利用率、显存利用率、PCIe RX/TX 吞吐、功耗、温度
- 基于 `psutil` 的可选主机遥测：CPU / 内存 / 磁盘 / 网络
- `Data`、`H2D`、`Fwd`、`Bwd`、`Opt`、`Eval`、`Ckpt`、`Log` 等阶段事件
- 聚合后的诊断摘要与瓶颈提示

典型安装与启动方式：

```bash
cd tools/trainflash_mcp
pip install -e .[mcp,host,test]
python -m trainflash_mcp
```

Hermes 宿主接入示例：

```yaml
mcp_servers:
  trainflash:
    command: "python"
    args: ["-m", "trainflash_mcp"]
    cwd: "/absolute/path/to/TrainFlashAgent/tools/trainflash_mcp"
    timeout: 180
    connect_timeout: 60
```

该 MCP server 暴露的工具包括：
- `get_trainflash_capabilities`
- `get_trainflash_system_snapshot`
- `start_trainflash_session`
- `record_trainflash_phase_event`
- `ingest_trainflash_phase_trace`
- `get_trainflash_summary`
- `stop_trainflash_session`

## Skills（核心工作流）
`/skills` 目录下的 skills 以英文编写，强调“契约 / 护栏 / 步骤 / 验收 / 报告模板”，便于 AI 稳定执行：
- [01-sandbox.md](skills/01-sandbox.md)：在任何修改前创建隔离沙盒
- [02-diagnose.md](skills/02-diagnose.md)：用低开销计时或 MCP 驱动的阶段诊断定位主要瓶颈
- [03-optimize.md](skills/03-optimize.md)：基于证据做工程与训练策略优化（effective batch、sampler/bucketing、训练环路开销等）
- [04-verify.md](skills/04-verify.md)：验证性能与保真；涉及训练策略变更时补充收敛感知检查；通过后再安全回写

## 为什么这套方法更稳
训练提速往往不是“某个算子慢”，而是系统层问题叠加：
- 数据管道与长尾 batch（I/O 抖动、样本耗时不均）
- padding 浪费 / shape 不稳定
- CPU↔GPU 同步点、过重的 logging / eval / checkpoint
- optimizer-step 策略（micro-batch、梯度累积、world size）导致的效率与收敛联动

这套 skills 的核心价值是：用统一流程把这些问题变成可测量、可复现、可审计的工程任务。

## 默认验收标准（可按项目调整）
- 性能：在目标 workload 上有可度量的提升（同时报告均值与尾部，如 p95/max）。
- 保真：在约定容忍度内（默认：mean loss delta < 1e-3；若项目有更合适指标，以项目为准）。
- 安全：未经明确确认，不得修改或回写原项目目录。

## 目录结构
```
TrainFlashAgent/
├── skills/                         # Playbooks (skills)
│   ├── 01-sandbox.md
│   ├── 02-diagnose.md
│   ├── 03-optimize.md
│   └── 04-verify.md
├── tools/
│   └── trainflash_mcp/         # 独立 MCP 诊断工具（遥测 + 阶段计时）
├── .cursorrules                     # Cursor 配置
├── .github/copilot-instructions.md  # Copilot 配置
├── README.md                        # English
└── README_zh.md                     # 中文
```

## License
MIT License

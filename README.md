# TrainFlashAgent ⚡

**让 AI 智能体像顶级性能工程师一样优化你的深度学习训练速度。**

TrainFlashAgent 是一个基于沙盒的、自顶向下的训练性能自动优化框架。它通过 LLM 驱动的闭环决策，在隔离环境中诊断瓶颈、实施工程级优化、验证效果，最终安全回写优化代码。

---

## 🚀 核心特性

- **🔒 沙盒隔离**：所有实验在物理隔离的沙盒中进行，确保原项目零风险
- **📊 自顶向下诊断**：优先解决 IO/数据长尾/同步阻塞等高影响瓶颈，而非陷入算子级微优化
- **🤖 Agent-Native**：设计为 MCP Server，可无缝集成到 Cursor/Claude Desktop 等 AI 编辑器
- **📈 量化审计**：每次修改都有明确的收益记录和模型保真度验证
- **🎯 晋级门控**：只有当前阶段达到性能平台期，才会解锁下一层级的诊断

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Server Layer                      │
│         (Expose skills to LLM via MCP protocol)          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  TuningManager                           │
│              (Orchestration & State Machine)             │
└──┬─────────┬────────────┬────────────┬──────────┬───────┘
   │         │            │            │          │
┌──▼───┐ ┌──▼────┐  ┌────▼─────┐ ┌───▼─────┐ ┌──▼────────┐
│Sandbox│ │Diagnostics│ │Intervention│ │Verification│ │Governance│
│Manager│ │(Top-Down)│ │(Engineering)│ │(Fidelity) │ │(Gate)   │
└───────┘ └─────────┘  └───────────┘ └──────────┘ └─────────┘
```

---

## 📋 优化流程

### Phase 1: 宏观诊断 (Macro-Diagnostics)
通过 `AST` 在训练循环的关键节点注入手动计时器，识别：
- 哪个阶段（DataLoad/Forward/Backward）是主要瓶颈
- 是否存在数据长尾导致的时间方差

### Phase 2: 工程调优 (Engineering Tuning)
根据诊断结果实施优化：
- **IO/Data**：调整 `num_workers`, `prefetch_factor`, 数据打包策略
- **资源**：优化 Batch Size、显存布局、梯度累积
- **等价替换**：将低效代码模式替换为高性能实现

### Phase 3: 微观诊断 (Micro-Diagnostics)
仅当 Phase 2 收益趋近于零时，调用 `torch.profiler` 进行算子级分析。

### 验证与回写
- **吞吐量基准**：测量 samples/sec 提升
- **模型保真度**：对比 Loss 曲线和梯度分布
- **用户确认**：收益报告 → 用户批准 → 安全回写

---

## 🛠️ 快速开始

### 安装
```bash
pip install -e .
```

### 作为 Python 库使用
```python
from trainflashagent import TuningManager

manager = TuningManager(
    project_root="./my_training_project",
    sandbox_root="./sandbox",
    baseline_throughput=100.0  # 可选：原始吞吐量
)

# 1. 初始化沙盒
manager.setup_environment()

# 2. 运行宏观诊断
report = manager.run_macro_diagnostic(
    target_file="train.py",
    target_func="train_step",
    label="TotalLoop"
)

# 3. 应用优化并验证
result = manager.apply_and_verify_intervention(
    file_rel_path="train.py",
    intervention_type="data_pipeline",
    params={"strategy": "balanced"}
)

# 4. 查看收益报告
print(manager.get_gain_report())

# 5. 用户批准后提交
manager.approve_pending_gains()

# 6. 最终合并
manager.finalize_and_merge()
```

### 作为 MCP Server 使用
```bash
python -m trainflashagent_mcp.server
```

在 Cursor 或 Claude Desktop 的 MCP 配置中添加：
```json
{
  "mcpServers": {
    "trainflashagent": {
      "command": "python",
      "args": ["-m", "trainflashagent_mcp.server"]
    }
  }
}
```

然后直接对 AI 说：
> "帮我把这个训练项目的速度提升 20%，不要影响精度"

---

## 📁 项目结构

```
TrainFlashAgent/
└── src/trainflashagent/
    ├── sandbox.py          # 沙盒隔离与版本管理
    ├── diagnostics.py      # 自顶向下诊断工具
    ├── interventions.py    # 工程级优化干预
    ├── verification.py     # 性能与保真度验证
    ├── governance.py       # 审计日志与晋级门控
    └── manager.py          # 统一编排器
```

---

## 🎯 适用场景

- ✅ 深度学习训练速度慢，但不知道瓶颈在哪
- ✅ 希望自动化调优流程，减少手动试错
- ✅ 需要确保优化不影响模型精度
- ✅ 使用 Cursor/Claude Desktop 等 AI 编辑器

---

## 📄 License

MIT

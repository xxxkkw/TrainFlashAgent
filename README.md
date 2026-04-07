# TrainFlashAgent ⚡

**让 AI 智能体像顶级性能工程师一样优化你的深度学习训练速度。**

TrainFlashAgent 是一套 MCP Skills，让 LLM 能够自主进行深度学习训练性能的自顶向下优化。LLM 通过调用这些 Skills，在沙盒环境中诊断瓶颈、实施优化、验证效果，最终安全回写。

---

## 🚀 核心特性

- **🔒 沙盒隔离**：所有实验在物理隔离的沙盒中进行，确保原项目零风险
- **📊 自顶向下诊断**：优先解决 IO/数据长尾/同步阻塞等高影响瓶颈
- **🤖 Agent-Native**：纯 Skills 设计，LLM 自主决策，无预设流程限制
- **🔧 模块化工具**：每个 Skill 独立，可自由组合调用

---

## 🏗️ 架构

```
LLM (自主决策)
     ↓ 调用 Skills
┌─────────────────────────────────────────┐
│         MCP Server (Skills Layer)        │
│  create_sandbox()  inject_timer()        │
│  collect_logs()    analyze_variance()    │
│  optimize_data_pipeline()  benchmark()   │
│  edit_code()       merge_to_main()       │
└────────────────────┬─────────────────────┘
                     ↓ 工具实现
           src/trainflashagent/
```

---

## 📋 Skills 列表

### 沙盒管理
| Skill | 说明 |
|-------|------|
| `create_sandbox(project_root, sandbox_name)` | 克隆项目到沙盒 |
| `create_snapshot(sandbox_name, label)` | 创建快照（修改前调用） |
| `rollback_to_snapshot(sandbox_name, label)` | 回滚到快照 |
| `merge_to_main(sandbox_name, project_root)` | 合并回原项目 |

### 诊断（Phase 1-3）
| Skill | 说明 |
|-------|------|
| `inject_timer(file_path, function, label)` | AST 注入手动计时器 |
| `collect_logs(script_path, steps)` | 运行并收集计时日志 |
| `analyze_variance(logs)` | 分析瓶颈和长尾 |
| `run_profiler(script_path, steps)` | 算子级 Profiler（最后手段） |

### 干预（Phase 2）
| Skill | 说明 |
|-------|------|
| `optimize_data_pipeline(file_path, strategy)` | 优化 DataLoader 设置 |
| `tune_resources(file_path, params)` | 调整 batch_size 等资源参数 |
| `edit_code(file_path, pattern, replacement)` | 通用代码修改 |

### 验证
| Skill | 说明 |
|-------|------|
| `benchmark_throughput(script_path, steps)` | 测量吞吐量 |
| `verify_fidelity(baseline_losses, optimized_losses)` | 验证模型保真度 |

### 辅助
| Skill | 说明 |
|-------|------|
| `read_file(file_path)` | 读取文件内容 |
| `list_files()` | 列出所有 Python 文件 |

---

## 🛠️ 快速开始

### 启动 MCP Server
```bash
python -m trainflashagent_mcp.server
```

### 在 Cursor / Claude Desktop 中配置
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

### 使用示例

配置完成后，直接对 AI 说：

> "帮我把 /path/to/my/training/project 的训练速度优化一下"

AI 会自动调用 Skills 执行以下流程：

1. **创建沙盒**：`create_sandbox("/path/to/my/training/project")`
2. **诊断瓶颈**：`inject_timer()` → `collect_logs()` → `analyze_variance()`
3. **实施优化**：根据诊断结果调用 `optimize_data_pipeline()` 或 `tune_resources()`
4. **验证效果**：`benchmark_throughput()` 对比优化前后
5. **回写代码**：`merge_to_main()`

---

## 📁 项目结构

```
TrainFlashAgent/
└── src/trainflashagent/
    ├── sandbox.py          # 沙盒工具
    ├── diagnostics.py      # 诊断工具
    ├── interventions.py    # 干预工具
    └── verification.py     # 验证工具
└── src/trainflashagent_mcp/
    └── server.py           # MCP Skills 暴露层
```

---

## 🎯 自顶向下方法论

1. **Phase 1 - 宏观诊断**：先用手动计时器识别哪个阶段是瓶颈
2. **Phase 2 - 工程调优**：优化 IO、数据加载、资源分配
3. **Phase 3 - 微观诊断**：仅当 Phase 2 收益趋近于零时，才调用 Profiler

---

## 📄 License

MIT

# TrainFlashAgent

<div align="center">

**A playbook for speeding up deep learning training: measure bottlenecks, apply engineering + training-policy fixes, and verify results.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cursor](https://img.shields.io/badge/Cursor-Supported-brightgreen)](https://cursor.sh)
[![Copilot](https://img.shields.io/badge/Copilot-Supported-blue)](https://github.com/features/copilot)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Supported-purple)](https://claude.ai/code)

</div>

English | [中文](README_zh.md)

## Overview
TrainFlashAgent is a set of Skills (Markdown playbooks) for improving deep learning training performance.
It is designed to be read and executed by an AI coding assistant inside your editor.

## Focus
- Core workflow: diagnose with low-overhead timing → optimize engineering + training policy → verify performance and fidelity → write back with approval.
- Primary targets: input pipeline stalls, long-tail batches, padding waste/shape instability, sync points, logging/eval/checkpoint overhead, optimizer-step policy.
- Optional accelerators: AMP / `torch.compile` / TF32 can be layered on later, but the skills here aim to find and fix the underlying system-level bottlenecks first.

## Design Principles
- Evidence first: prove the bottleneck before tuning.
- Top-down workflow: macro timing → engineering/training-policy fixes → (profiler only if needed).
- Safety by default: work in a sandbox; write back only after verification and explicit approval.
- Reviewable changes: plan with files, change one variable per experiment, keep a tuning log.

## Quick Start
1) Copy instructions into the training project you want to optimize:

```bash
# Cursor users
cp .cursorrules /path/to/your/project/

# GitHub Copilot users
mkdir -p /path/to/your/project/.github
cp .github/copilot-instructions.md /path/to/your/project/.github/

# Claude Code users
cp CLAUDE.md /path/to/your/project/
```

2) Ask your AI assistant to follow the skills:

> “Speed up training in this repo. Follow TrainFlashAgent skills strictly and report measurable results.”

Expected flow: sandbox → diagnose → optimize → verify → (optional) write back.

## MCP-based diagnosis helper
TrainFlashAgent now includes a standalone TrainFlash MCP package at `tools/trainflash_mcp`.

It is intended for low-overhead training diagnosis before any heavy profiler run, and can combine:
- GPU telemetry from NVML (`gpu_util`, memory utilization, PCIe RX/TX throughput, power, temperature)
- optional host telemetry from `psutil` (CPU / memory / disk / network)
- phase timing events for `Data`, `H2D`, `Fwd`, `Bwd`, `Opt`, `Eval`, `Ckpt`, `Log`
- aggregated diagnosis summaries and bottleneck hints

Typical setup:

```bash
cd tools/trainflash_mcp
pip install -e .[mcp,host,test]
python -m trainflash_mcp
```

Hermes host config example:

```yaml
mcp_servers:
  trainflash:
    command: "python"
    args: ["-m", "trainflash_mcp"]
    cwd: "/absolute/path/to/TrainFlashAgent/tools/trainflash_mcp"
    timeout: 180
    connect_timeout: 60
```

The MCP server exposes:
- `get_trainflash_capabilities`
- `get_trainflash_system_snapshot`
- `start_trainflash_session`
- `record_trainflash_phase_event`
- `ingest_trainflash_phase_trace`
- `get_trainflash_summary`
- `stop_trainflash_session`

## The Skills
- [01-sandbox.md](skills/01-sandbox.md): isolate a sandbox workspace before any edits
- [02-diagnose.md](skills/02-diagnose.md): identify the dominant bottleneck with low-overhead timing or MCP-driven phase diagnosis
- [03-optimize.md](skills/03-optimize.md): apply engineering + training-strategy optimizations (effective batch, sampler/bucketing, loop overheads)
- [04-verify.md](skills/04-verify.md): verify performance + fidelity (and convergence sanity when needed), then write back safely

## Acceptance Criteria (Default)
- Performance: measurable improvement on the target workload (report mean and tail, not just one number).
- Fidelity: within the agreed tolerance (default: mean loss delta < 1e-3, unless the project defines a better metric).
- Safety: no writeback to the original project directory without explicit user approval.

## Why This Approach Works
Training performance issues are usually systems issues:
- input pipeline stalls and long-tail batches
- padding waste / shape instability
- sync points and logging overhead
- optimizer-step policy (micro-batch vs accumulation vs world size)

These require measurement, iteration discipline, and careful verification—exactly what the skills encode.

## Repository Layout
```
TrainFlashAgent/
├── skills/                         # Playbooks (skills)
│   ├── 01-sandbox.md
│   ├── 02-diagnose.md
│   ├── 03-optimize.md
│   └── 04-verify.md
├── tools/
│   └── trainflash_mcp/         # Standalone MCP helper for telemetry + phase timing
├── .cursorrules                     # Cursor instructions
├── .github/copilot-instructions.md  # Copilot instructions
├── README.md                        # English (this file)
└── README_zh.md                     # 中文说明
```

## License
MIT License

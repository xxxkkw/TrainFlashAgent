# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.在cli中回答时默认保持中文

## Project: TrainFlashAgent
An AI-driven, top-down performance optimizer for deep learning training. It focuses on engineering-level bottlenecks (IO, data skew, synchronization) using a sandboxed iteration loop.

## Development Commands
- **Run Tests**: `pytest tests/`
- **Run Single Test**: `pytest tests/test_file.py::test_name`
- **Linting**: `flake8 src/` (if installed)
- **MCP Server**: `python src/trainflashagent_mcp/server.py`

## Architecture & Structure
The project follows an Agent-native "Skill-based" architecture:

- `docs/superpowers/specs/`: Contains the core design specifications.
- `docs/superpowers/plans/`: Contains the implementation plans.
- `src/trainflashagent/`: The core SDK.
    - `sandbox.py`: Project cloning, snapshotting, and merging.
    - `diagnostics.py`: Top-down probing (Manual timers $\rightarrow$ Profilers).
    - `interventions.py`: Engineering fixes (Data pipeline, Resource config).
    - `verification.py`: Throughput benchmarks and model fidelity (Loss/Gradient) checks.
    - `governance.py`: State machine for phase progression and tuning logs.
- `src/trainflashagent_mcp/`: MCP server wrapper exposing the SDK as tools for LLMs.

## Core Workflow
The Agent must operate according to the **Top-Down Methodology**:
1. **Sandbox Clone** $\rightarrow$ 2. **Macro-Diagnostics (Manual Timers)** $\rightarrow$ 3. **Engineering Tuning** $\rightarrow$ 4. **Verify Fidelity & Speed** $\rightarrow$ 5. **Progression Gate** (Prove plateau) $\rightarrow$ 6. **Micro-Diagnostics (Profiler)** $\rightarrow$ 7. **Merge to Main**.

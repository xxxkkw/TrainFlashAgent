# TrainFlashAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sandboxed, top-down performance optimization agent for DL training that prioritizes engineering fixes over kernel tuning.

**Architecture:** A modular Python SDK providing a set of "Skills" (Sandbox, Macro-Diagnostics, Engineering Tuning, Verification) managed by a Governance state machine.

**Tech Stack:** Python 3.10+, PyTorch, MCP SDK, `ast` (for code injection), `time.perf_counter`.

---

## Phase 1: Sandbox & Foundation

### Task 1: Sandbox Manager Implementation
**Files:**
- Create: `src/trainflashagent/sandbox.py`
- Test: `tests/test_sandbox.py`

- [ ] **Step 1: Write failing test for `sandbox_clone`**
```python
def test_sandbox_clone():
    # Setup dummy project
    os.makedirs("src_proj", exist_ok=True)
    with open("src_proj/train.py", "w") as f: f.write("print('hello')")
    
    from trainflashagent.sandbox import SandboxManager
    sm = SandboxManager()
    sm.sandbox_clone("src_proj", "dst_proj")
    assert os.path.exists("dst_proj/train.py")
```

- [ ] **Step 2: Implement `sandbox_clone` using `shutil.copytree`**
```python
import shutil
import os

class SandboxManager:
    def sandbox_clone(self, src, dst):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
```

- [ ] **Step 3: Implement `sandbox_snapshot` and `sandbox_rollback`**
- [ ] **Step 4: Implement `merge_to_main` (selective file copy)**
- [ ] **Step 5: Run tests and commit**

---

## Phase 2: Macro-Diagnostics (The Top)

### Task 2: Manual Timer Injection
**Files:**
- Create: `src/trainflashagent/diagnostics.py`
- Test: `tests/test_diagnostics.py`

- [ ] **Step 1: Write failing test for `inject_manual_timer`**
```python
def test_inject_timer():
    code = "def train():\n    print('start')\n    print('end')"
    from trainflashagent.diagnostics import TimerInjector
    ti = TimerInjector()
    # Inject at start of train()
    updated = ti.inject_manual_timer(code, "train", "TotalLoop")
    assert "time.perf_counter()" in updated
```

- [ ] **Step 2: Implement `TimerInjector` using `ast` module**
- [ ] **Step 3: Implement `collect_timer_logs`** (Run script $\rightarrow$ parse output $\rightarrow$ return JSON)
- [ ] **Step 4: Implement `analyze_time_variance`** (Calculate mean, std, and max for long-tail detection)
- [ ] **Step 5: Run tests and commit**

---

## Phase 3: Engineering Intervention (The Middle)

### Task 3: Resource & Data Tuning
**Files:**
- Create: `src/trainflashagent/interventions.py`

- [ ] **Step 1: Implement `optimize_data_pipeline`** (Logic to find and update `num_workers`, `prefetch_factor` in common PyTorch patterns)
- [ ] **Step 2: Implement `tune_resource_config`** (Logic to adjust `batch_size` or `grad_accum` in config files/args)
- [ ] **Step 3: Implement `apply_equivalent_substitution`** (Basic mapping of slow patterns to fast ones, e.g., replacing explicit loops with `torch.stack`)
- [ ] **Step 4: Run integration tests and commit**

---

## Phase 4: Verification & Fidelity

### Task 4: Performance & Fidelity Suite
**Files:**
- Create: `src/trainflashagent/verification.py`
- Test: `tests/test_verification.py`

- [ ] **Step 1: Implement `benchmark_throughput`** (Measure samples/sec over $N$ steps)
- [ ] **Step 2: Implement `verify_model_fidelity`**
    - Compare Loss values of baseline vs sandbox.
    - Check Gradient Norms.
- [ ] **Step 3: Run tests and commit**

---

## Phase 5: Governance & State Machine

### Task 5: Progression Gate & Tuning Log
**Files:**
- Create: `src/trainflashagent/governance.py`
- Modify: `src/trainflashagent/manager.py` (Coordination)

- [ ] **Step 1: Implement `TuningLog` class** (Record Modification $\rightarrow$ Gain $\rightarrow$ Status)
- [ ] **Step 2: Implement `ProgressionGate`** (Logic: If Gain << Threshold Threshold for 3 attempts $\rightarrow$ Unlock next phase)
- [ ] **Step 3: Implement `TuningManager`** (The high-level API: `start_optimization()`, `request_merge()`)
- [ ] **Step 4: Run end-to-end integration test and commit**

---

## Phase 6: MCP Server Integration

### Task 6: MCP Server Wrapper
**Files:**
- Create: `src/trainflashagent_mcp/server.py`

- [ ] **Step 1: Setup MCP Server boilerplate**
- [ ] **Step 2: Map `TuningManager` methods to MCP Tools**
- [ ] **Step 3: Verify tool execution via `mcp-inspector`**
- [ ] **Step 4: Final Commit**

# TrainFlashAgent - GitHub Copilot Instructions

## Overview

TrainFlashAgent is a methodology for optimizing deep learning training performance using a top-down approach. When helping users optimize training code, follow this methodology.

---

## Optimization Methodology

### Phase 1: Create Sandbox
Always work in an isolated sandbox first:
```bash
rsync -av /path/to/project/ /tmp/trainflash_sandbox/
```

### Phase 2: Macro Diagnostics With TrainFlash MCP (NOT Profiler)
Prefer `tools/trainflash_mcp` first:
```bash
cd tools/trainflash_mcp
pip install -e .[mcp,host,test]
python -m trainflash_mcp
```

Default MCP call sequence:
- `get_trainflash_capabilities`
- `start_trainflash_session`
- `record_trainflash_phase_event` or `ingest_trainflash_phase_trace`
- `get_trainflash_system_snapshot`
- `get_trainflash_summary`
- `stop_trainflash_session`

Only fall back to manual timers when MCP cannot be installed or connected.

Fallback manual timers at key boundaries:
```python
import time

# At DataLoader boundary
start = time.perf_counter()
batch = next(iter(dataloader))
print(f"[TFA_TIMER] DataLoader {time.perf_counter() - start:.6f}")

# At Forward boundary
start = time.perf_counter()
output = model(batch)
print(f"[TFA_TIMER] Fwd {time.perf_counter() - start:.6f}")
```

Run 50-100 steps and analyze:
- Which phase takes most time?
- Is there long-tail (max >> mean)?

### Phase 3: Engineering Optimizations
Based on diagnostics:

**If DataLoader is bottleneck (>40% time):**
```python
DataLoader(dataset, 
    num_workers=8,           # Increase from 0
    prefetch_factor=4,       # Add prefetching
    persistent_workers=True,
    pin_memory=True
)
```

**If compute dominates but utilization is low or variance is high:**
- Adjust effective batch strategy (micro-batch vs gradient accumulation) and document optimizer-step cadence.
- Use sampler/bucketing or shape stabilization for variable-size workloads to reduce padding waste and tail latency.
- Reduce training-loop overhead (logging / eval / checkpoint cadence) during tuning runs.

### Phase 4: Verify
- Measure throughput improvement
- Verify loss curve consistency (diff < 0.001)
- If training strategy changed, add a short convergence-aware sanity check under an explicit “equal budget” protocol
- Get user approval before merging

---

## Key Principles

1. **Top-Down**: TrainFlash MCP FIRST, fallback manual timers SECOND, Profiler LAST
2. **Sandbox First**: Never modify original code directly
3. **Verify Fidelity**: Ensure model accuracy is preserved
4. **User Approval**: Always report before merging changes

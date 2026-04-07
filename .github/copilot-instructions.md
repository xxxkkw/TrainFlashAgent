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

### Phase 2: Macro Diagnostics (NOT Profiler)
Insert manual timers at key boundaries:
```python
import time

# At DataLoader boundary
start = time.perf_counter()
batch = next(iter(dataloader))
print(f"[TRAINOPT_TIMER] DataLoader: {time.perf_counter() - start:.4f}s")

# At Forward boundary  
start = time.perf_counter()
output = model(batch)
print(f"[TRAINOPT_TIMER] Forward: {time.perf_counter() - start:.4f}s")
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

**If GPU utilization is low (<70%):**
```python
# Increase batch_size
# Or enable mixed precision
from torch.cuda.amp import autocast, GradScaler
```

### Phase 4: Verify
- Measure throughput improvement
- Verify loss curve consistency (diff < 0.001)
- Get user approval before merging

---

## Key Principles

1. **Top-Down**: Manual timers FIRST, Profiler LAST
2. **Sandbox First**: Never modify original code directly
3. **Verify Fidelity**: Ensure model accuracy is preserved
4. **User Approval**: Always report before merging changes

# Skill 02 — Diagnose (Top-Down Bottleneck Identification)

## Purpose
- Identify the dominant training bottleneck with low overhead.
- Start macro-first (input pipeline vs compute vs synchronization) before using heavy profilers.

## Scope
- Applies to: any training loop where you can add lightweight timing.
- Not in scope: kernel-level tuning as the first step.

## Principle
- MUST start with manual timing (step-level phases).
- MAY use a profiler only after engineering-level optimizations plateau.

## Contract
**Inputs**
- Training entry point (script/command) and a way to run a fixed number of steps.
- A representative dataset slice (or a deterministic synthetic loader if the real dataset is too slow to iterate).

**Outputs**
- A “Diagnostic Report” with:
  - Mean step-time breakdown by phase (percentages + seconds)
  - Variance / long-tail signals (p95/max vs mean)
  - A single primary bottleneck hypothesis and next action focus

## Guardrails (MUST)
- MUST keep instrumentation easy to remove (single helper + tag prefix).
- MUST include warmup steps and then measure on a fixed window.
- If measuring GPU work, MUST avoid misleading async timing:
  - Option A (recommended): synchronize before stopping a timer.
  - Option B: time CPU-side only and label it as such.

## Procedure
### Step 1 — Define phases to time
Time at least these phases per training step:
- `Data`: waiting for the next batch / preprocessing
- `H2D`: host-to-device transfer (if explicit)
- `Fwd`: forward pass (+ loss)
- `Bwd`: backward pass
- `Opt`: optimizer step (+ zero grad)

If your training loop includes significant “outside-step” work, also time:
- `Eval`: validation / metrics runs
- `Ckpt`: checkpoint saving
- `Log`: heavy logging / visualization

### Step 2 — Add a minimal timer utility
Python (stdlib-only):
```python
import time

class PhaseTimer:
    def __init__(self, prefix="[TFA_TIMER]"):
        self.prefix = prefix
        self.t0 = None
        self.name = None

    def start(self, name: str):
        self.name = name
        self.t0 = time.perf_counter()

    def stop(self, extra: str = ""):
        dt = time.perf_counter() - self.t0
        if extra:
            print(f"{self.prefix} {self.name} {dt:.6f} {extra}")
        else:
            print(f"{self.prefix} {self.name} {dt:.6f}")
        self.t0 = None
        self.name = None
```

If you use PyTorch and want accurate GPU timings, synchronize right before `stop()`:
```python
def cuda_sync_if_available():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
```

### Step 3 — Instrument the training step
Pseudo-structure (adapt to your project):
```python
timer = PhaseTimer()

for step_idx, batch in enumerate(dataloader):
    if step_idx < WARMUP_STEPS:
        pass

    timer.start("Data")
    # batch is already fetched by the loop; if you can time fetch explicitly, do it there
    timer.stop()

    timer.start("H2D")
    # move tensors to GPU if applicable
    # cuda_sync_if_available()
    timer.stop()

    timer.start("Fwd")
    # forward + loss
    # cuda_sync_if_available()
    timer.stop()

    timer.start("Bwd")
    # backward
    # cuda_sync_if_available()
    timer.stop()

    timer.start("Opt")
    # optimizer step + zero_grad
    # cuda_sync_if_available()
    timer.stop()
```

### Step 4 — Run a controlled measurement
- Warm up `WARMUP_STEPS` (e.g., 10–20).
- Measure `MEASURE_STEPS` (e.g., 50–200).
- Save logs to a file.

### Step 5 — Summarize (stdlib-only)
```python
import re
import statistics

pattern = re.compile(r"^\[TFA_TIMER\]\s+(\w+)\s+([0-9.]+)", re.M)
text = open("timer_logs.txt", "r", encoding="utf-8", errors="ignore").read()
rows = pattern.findall(text)

by_phase = {}
for name, value in rows:
    by_phase.setdefault(name, []).append(float(value))

def pct(x, total):
    return 100.0 * x / total if total > 0 else 0.0

means = {k: statistics.mean(v) for k, v in by_phase.items() if v}
step_mean = sum(means.values())

for k in sorted(means, key=lambda n: means[n], reverse=True):
    v = by_phase[k]
    p95 = statistics.quantiles(v, n=20)[-1] if len(v) >= 20 else max(v)
    print(f"{k:>4} mean={means[k]:.6f}s  p95={p95:.6f}s  share={pct(means[k], step_mean):5.1f}%")
```

## Interpretation Cheatsheet
- If `Data` dominates: input pipeline / dataloader is the primary bottleneck.
- If `Fwd+Bwd+Opt` dominates but GPU utilization is low: batch sizing, mixed precision, or sync points likely.
- If `p95` or `max` is much larger than mean: long-tail stalls (I/O jitter, CPU contention, random heavy samples).

## Training-Strategy Signals (Optional)
These help when “samples/sec” is misleading for variable-size workloads.
- Track “work per step”:
  - tokens/frames/pixels per step (depending on modality)
  - average padding ratio (if applicable)
- If step time correlates with “work per step”, consider bucketing/sampler changes in Skill 03.

## Diagnostic Report (Template)
```
[Diagnostic Report]
Workload: <model/dataset/steps>
Hardware: <CPU/GPU/Storage>
Window: warmup=<WARMUP_STEPS>, measure=<MEASURE_STEPS>

Step Time Breakdown (mean):
- Data: <sec> (<%>)
- H2D:  <sec> (<%>)
- Fwd:  <sec> (<%>)
- Bwd:  <sec> (<%>)
- Opt:  <sec> (<%>)

Tail Signals:
- Long-tail phase: <phase or none>
- Evidence: p95/mean=<x>, max/mean=<y>

Primary Bottleneck:
- <one sentence>

Next Focus:
- Proceed to Skill 03 — Optimize with focus on <data|compute|sync>
```

# Skill 03 — Optimize (Engineering-Level Speedups)

## Purpose
- Apply the smallest set of changes that materially improves training throughput/step-time.
- Prefer “engineering wins” (input pipeline, batching strategy, synchronization removal, training-loop policy) before kernel-level tuning.

## Contract
**Inputs**
- A completed Diagnostic Report from Skill 02.
- Preferably the latest TrainFlash MCP outputs as well: `get_trainflash_summary`, `get_trainflash_system_snapshot`, and `get_trainflash_capabilities`.
- Constraints: maximum acceptable quality drift, allowed code change scope, and whether new dependencies are allowed.

**Outputs**
- An Optimization Plan: ordered list of candidate changes with expected impact and risk.
- A measurable before/after result (throughput or step-time).
- A short Optimization Log that can be used for review and later rollback.

## Guardrails (MUST)
- MUST change one variable at a time when benchmarking (avoid confounded wins).
- MUST preserve baseline behavior unless explicitly documented (same data order/seed if applicable).
- MUST keep changes reversible and reviewable (small diff, clear config knobs).
- MUST NOT introduce new dependencies without explicitly stating why and what alternatives exist.
- MUST keep “speed” and “training efficiency” separate:
  - Step throughput improvements are not sufficient if they worsen convergence (see Skill 04).

## Prioritization (Default)
1. Input pipeline (I/O, dataloader parallelism, tail latency)
2. Training strategy (effective batch, sampler/bucketing, shape stabilization)
3. Resource utilization (batching, gradient accumulation)
4. Synchronization removal (CPU↔GPU stalls, logging, `.item()` patterns)
5. Kernel-level / profiler-driven micro-optimizations

## Decision Map (Start Here)
- If TrainFlash MCP says only `telemetry_only` is available, stay focused on macro bottlenecks and do not assume precise memcpy/kernel timelines exist.
- If `Data` is the largest share or shows long-tail: focus on **Input pipeline**.
- If compute dominates but GPU utilization is low: focus on **Batching / overlap / stalls**.
- If timings show large variance or unexpected stalls: focus on **Sync removal**.

## Planning With Files (Lightweight)
Before editing, produce a 5–10 line plan that is easy to review.

Template:
```
[Optimization Plan]
Hypothesis: <what is limiting throughput or training efficiency>

Files to Modify:
- <path>: <what change>

Benchmark Plan:
- Baseline: <command/config>, warmup=<n>, measure=<n>, metric=<samples/sec|sec/step|tokens/sec>
- Optimized: <same protocol>

Risks:
- <convergence drift / stability / tail latency / memory>
```

## Playbook
### Track A — Input Pipeline (Data dominates)
**Goal:** keep the GPU fed; reduce tail latency.

Common actions (PyTorch examples; adapt if your stack differs):
- Increase dataloader parallelism:
  - `num_workers`: start from `min(8, cpu_cores)` and tune.
  - `persistent_workers=True` (when workers are >0).
  - `prefetch_factor`: start at 2–4; increase if GPU waits for data.
  - `pin_memory=True` (if using CUDA).

Example:
```python
import os

cpu_cores = os.cpu_count() or 8
num_workers = min(8, cpu_cores)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    prefetch_factor=4,
    persistent_workers=(num_workers > 0),
    pin_memory=True,
)
```

If long-tail is present (p95/max ≫ mean):
- Identify whether tail comes from:
  - slow samples (heavy decoding / variable-length padding)
  - shared storage jitter
  - CPU contention (too many workers, other processes)
- Mitigations:
  - move expensive preprocessing offline or cache results
  - reduce per-sample Python work in `__getitem__`
  - bucket by sequence length / use dynamic padding (NLP/audio) to reduce worst-case batches
  - tune workers downward if the system thrashes

### Track B — Effective Batch & Optimizer-Step Policy (Training strategy)
**Goal:** increase efficiency per wall-clock hour without breaking convergence.

Definitions:
- `micro_batch`: per-device batch processed per forward/backward
- `accum_steps`: number of micro-batches per optimizer update
- `effective_batch`: `micro_batch * accum_steps * world_size` (if distributed)

Rules of thumb (MUST document which one you use):
- If you change `effective_batch`, you MUST explicitly state how learning rate and scheduler stepping behave.
- Compare results under a consistent “budget”:
  - same number of optimizer steps, or
  - same number of samples/tokens, or
  - same wall-clock (for throughput-only checks)

Minimal checklist:
- [ ] `optimizer.step()` frequency matches the intended `accum_steps`
- [ ] scheduler steps on the intended cadence (per optimizer step vs per micro-step)
- [ ] logging reports both micro-step and optimizer-step counts

### Track C — Sampler / Bucketing / Shape Stabilization (Training strategy)
**Goal:** reduce padding waste and long-tail batches; keep shapes stable.

Use when:
- sequences/images have variable length/shape
- `Data` shows long-tail, or step time variance is high
- average throughput is fine but p95/max is bad

Actions (pick the smallest viable):
- Length/shape bucketing (group similar lengths/shapes into the same batch)
- “Sortish” sampling (partial sort + shuffle to keep randomness)
- Cap extreme samples (only if acceptable; MUST document impact)
- Track “work per step”:
  - tokens/frames/pixels per step, not only samples per step

Acceptance signal:
- step time p95 drops meaningfully, not just the mean

### Track D — Training Loop Overheads (Eval / Checkpoint / Logging policy)
**Goal:** reduce non-training work that steals wall-clock time.

Common problems:
- validation too frequent
- checkpoint saving too frequent or too large
- per-step logging/metrics doing heavy CPU work

Actions:
- Decouple “benchmark config” from “full training config”
- Log every N steps; aggregate metrics on-device when possible; sync less often
- Run eval less frequently during exploratory tuning, then restore for final runs
- Save checkpoints on a coarser schedule (or only on improvements) during tuning

### Track E — Distributed Training Policy (If applicable)
**Goal:** avoid synchronization overheads and correctness traps in multi-GPU training.

Actions:
- Use gradient accumulation with `no_sync()` (DDP) for micro-steps, sync only on the final micro-step
- Ensure distributed samplers are configured correctly (e.g., epoch setting, sharding)
- Ensure eval/checkpointing happens on rank 0 only

Guardrails:
- MUST verify `effective_batch` calculation includes `world_size`
- MUST verify reproducibility/seed behavior is acceptable for comparisons

### Track F — Resource Utilization (Compute dominates, GPU underutilized)
**Goal:** increase effective work per step and reduce overhead.

Actions:
- Increase `batch_size` until memory is near the target utilization.
- If OOM, use gradient accumulation to raise effective batch size.
- Use mixed precision (if supported) and validate quality.

Gradient accumulation sketch:
```python
accum_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

### Track G — Synchronization Removal (Stalls / high variance)
**Goal:** avoid accidental CPU↔GPU sync and reduce Python overhead.

Common fixes:
- Reduce frequent `.item()` calls inside the step loop.
- Avoid per-step heavy logging/printing; log every N steps.
- Keep metrics aggregation on-device when possible; only sync occasionally.
- Ensure validation runs under no-grad/inference mode if applicable.

## Benchmark Protocol (MUST)
- Use the same command-line/config for baseline and optimized runs.
- Include warmup, then measure on a fixed step window.
- Record:
  - throughput (samples/sec) or step time (sec/step)
  - max memory usage (if available)
  - any stability issues (OOM, NaNs, divergence)

## Definition of Done
- [ ] One or more changes are applied with a clean, reviewable diff.
- [ ] Performance improves by a meaningful amount for the target workload.
- [ ] No quality regression beyond the agreed tolerance.
- [ ] A complete Optimization Log is produced.

## Optimization Log (Template)
```
[Optimization Log]
Baseline:
- Workload: <model/dataset/steps>
- Metric: <samples/sec or sec/step>
- Result: <value>

Change Set:
- Files/Configs: <list>
- Change: <one sentence>
- Rationale: <one sentence linked to Diagnostic Report>

Optimized:
- Metric: <same metric>
- Result: <value>
- Delta: <+x%>

Quality / Fidelity:
- Criterion: <loss delta / eval metric / tolerance>
- Status: <pass|fail>

Next:
- Proceed to Skill 04 — Verify (full check + writeback decision)
```

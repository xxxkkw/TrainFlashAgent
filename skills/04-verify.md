# Skill 04 — Verify & Write Back (Prove It, Then Merge Safely)

## Purpose
- Confirm the optimization is real (performance) and safe (quality/fidelity).
- Produce a reviewable report and only then write changes back to the original project.

## Contract
**Inputs**
- Baseline run artifacts: logs/metrics for the target workload.
- Optimized run artifacts: logs/metrics collected under the same conditions.
- When TrainFlash MCP is available, include baseline/optimized `get_trainflash_summary` outputs and relevant `get_trainflash_system_snapshot` evidence.
- A defined tolerance for quality drift (default: mean loss delta < 1e-3, unless the project defines a better metric).

**Outputs**
- A Verification Report (performance + fidelity + risk notes).
- A writeback decision: approve / reject / need more testing.
- If training-strategy changes are involved: a short convergence-aware comparison (see Procedure).

## Guardrails (MUST)
- MUST compare apples-to-apples:
  - same code path (except the change set)
  - same dataset split and ordering (or document differences)
  - same number of warmup and measured steps
- MUST record the exact commands/config used for both runs.
- MUST wait for explicit approval before modifying the original project directory.
- If the change set alters training strategy (batching/sampler/optimizer-step cadence), MUST state what “equal budget” means:
  - same optimizer steps, or same samples/tokens, or same wall-clock.

## Procedure
### Step 1 — Establish a baseline and an optimized run
- Run baseline and optimized variants using the same benchmark protocol:
  - warmup window
  - measurement window
  - identical batch size unless the change set explicitly modifies it (then document it)

### Step 2 — Performance verification
Choose a primary metric and keep it consistent:
- Throughput: `samples/sec`
- Or step time: `sec/step`

When TrainFlash MCP is available, compare the phase-share and telemetry deltas between the baseline and optimized runs, not only the top-line throughput.

Minimum output:
- Baseline: mean + p95 (or max) over the measurement window
- Optimized: mean + p95 (or max)
- Delta: percentage improvement

If the gain is only on mean but tails worsen significantly, document it as a risk (tail regressions often matter in long training runs).

### Step 2.5 — Convergence-aware sanity check (when training strategy changes)
Use this when you change any of:
- effective batch (`micro_batch`, `accum_steps`, `world_size`)
- sampler / bucketing / shape policy
- eval/checkpoint cadence (can affect wall-clock comparisons)

Pick one protocol and state it explicitly:
- Protocol A: same number of **optimizer steps** (recommended for accumulation changes)
- Protocol B: same number of **samples/tokens**
- Protocol C: same **wall-clock** (only for pure throughput checks)

Minimum output:
- early loss slope over a short window (e.g., first N optimizer steps)
- any divergence signals (NaN/Inf, unstable loss)

### Step 3 — Fidelity / quality verification
Default checks (pick what is meaningful for the project):
- Loss drift:
  - Compare loss series on the same steps and compute mean/max absolute delta.
- Evaluation metric drift (preferred if available):
  - Compare validation accuracy/F1/bleu/etc. on a fixed evaluation set.

Stdlib-only loss extractor sketch (adapt your log format):
```python
import re
import statistics

def extract_losses(path: str):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    vals = re.findall(r"loss[:\s]+([0-9.]+)", text)
    return [float(v) for v in vals]

base = extract_losses("baseline.txt")
opt = extract_losses("optimized.txt")
paired = list(zip(base, opt))[: min(len(base), len(opt))]
diffs = [abs(a - b) for a, b in paired]

print("pairs:", len(paired))
print("mean_abs_delta:", statistics.mean(diffs) if diffs else None)
print("max_abs_delta:", max(diffs) if diffs else None)
```

If mixed precision or algorithmic changes are involved, tighten the protocol:
- fix random seeds and determinism flags where possible
- run a slightly longer window
- optionally add a small eval run after N steps

### Step 4 — Decision + writeback plan
- If performance improves and fidelity is within tolerance: recommend **Approve**.
- If fidelity fails: **Reject** and revert the change set in the sandbox.
- If results are noisy or environment differs: **More Testing** with a clearer protocol.

### Step 5 — Write back (only after approval)
Preferred approaches (pick one and document it):
- Copy only the changed files back.
- Apply a patch/diff in the original repo.
- Use a selective sync with explicit excludes.

## Definition of Done
- [ ] Verification Report is complete and reproducible.
- [ ] Performance outcome is clearly stated (mean + tail).
- [ ] Fidelity outcome is clearly stated with an explicit tolerance.
- [ ] Writeback is performed only after approval.

## Code Review Checklist (Before Writeback)
- Safety: changes were produced and validated in the sandbox.
- Scope: the diff is minimal and focused (avoid unrelated refactors).
- Attribution: one change set corresponds to one measured result (no confounded benchmarks).
- Evidence: baseline vs optimized metrics are attached (mean + tail).
- Fidelity: quality checks pass under the stated tolerance.
- Risk: any tail regressions, memory changes, or stability risks are documented.

## Verification Report (Template)
```
[Verification Report]
Workload: <model/dataset/steps>
Environment: <CPU/GPU/driver/storage>

Commands:
- Baseline: <command or config>
- Optimized: <command or config>

Performance (measurement window):
- Metric: <samples/sec or sec/step>
- Baseline: mean=<v>, p95=<v>
- Optimized: mean=<v>, p95=<v>
- Delta: <+x%>

Fidelity:
- Criterion: <loss delta / eval metric / tolerance>
- Result: <numbers>
- Status: <pass|fail>

Risk Notes:
- <tails, memory, stability, determinism, etc.>

Decision Required:
□ Approve (write back)
□ Reject (rollback in sandbox)
□ More Testing (specify what to run)
```

# trainflash_mcp

Lightweight MCP server for training diagnosis.

This package is intended to replace ad hoc timer scripts inside TrainFlashAgent's diagnose flow.
It combines:
- low-overhead GPU telemetry from NVML
- optional host telemetry from psutil
- runtime profiler capability discovery for `nsys` / `ncu` / `iostat` / `pidstat`
- explicit phase timing events for `Data`, `H2D`, `Fwd`, `Bwd`, `Opt`, `Eval`, `Ckpt`, `Log`
- aggregated summaries and bottleneck hints suitable for TrainFlashAgent reports

## What it does
- samples GPU core utilization, memory utilization, PCIe throughput proxy, power, temperature, GPU name, SM clock, memory clock
- exposes coarse H2D/D2H evidence through NVML PCIe TX/RX throughput counters
- samples host CPU / memory / disk / network when `psutil` is available
- detects whether precise timeline / memcpy profiling backends are present (`nsys`, `ncu`) and whether host/process IO CLIs are available (`iostat`, `pidstat`)
- records phase start/end events or ingests a JSONL trace emitted by a lightweight helper
- summarizes mean / p50 / p95 / max / share for phase timings
- produces simple diagnosis hints linking phase timing with hardware telemetry

## Installation

```bash
pip install -e .
pip install -e .[mcp]
# optional host telemetry
pip install -e .[host]
```

## Run as MCP server

```bash
trainflash-mcp
# or
python -m trainflash_mcp
```

## MCP tools
- `start_trainflash_session`
- `record_trainflash_phase_event`
- `ingest_trainflash_phase_trace`
- `get_trainflash_summary`
- `get_trainflash_system_snapshot`
- `stop_trainflash_session`
- `get_trainflash_capabilities`

`get_trainflash_system_snapshot` returns a live snapshot for GPU core utilization, coarse PCIe-based H2D/D2H pressure, host IO, and machine capability flags. The capability payload distinguishes:
- coarse transfer evidence available now via NVML (`coarse_h2d_d2h_supported`)
- precise memcpy / timeline support only when heavy profilers are installed (`precise_h2d_d2h_supported`, `precise_gpu_timeline_supported`)
- recommended collection mode (`telemetry_only` vs `telemetry_plus_profiler`)

## Minimal helper pattern inside a training repo

Emit JSONL events such as:

```json
{"step": 12, "phase": "Data", "event": "start", "ts": 1710000000.1}
{"step": 12, "phase": "Data", "event": "end", "ts": 1710000000.13}
{"step": 12, "phase": "H2D", "event": "start", "ts": 1710000000.13}
{"step": 12, "phase": "H2D", "event": "end", "ts": 1710000000.15}
```

Then import it into the running TrainFlash session with `ingest_trainflash_phase_trace`.

## Python usage

```python
from trainflash_mcp import TrainFlashMonitor

monitor = TrainFlashMonitor(interval_ms=100)
sid = monitor.start("train-short-window", metadata={"model": "demo"})
monitor.record_trainflash_phase_event(sid, phase="Data", event="start", step=0)
monitor.record_trainflash_phase_event(sid, phase="Data", event="end", step=0)
summary = monitor.stop(sid)
print(summary["phase_summary"])
```

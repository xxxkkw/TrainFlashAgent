from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean, median
from typing import Any


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _aggregate(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "avg": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": float(median(values)),
        "p95": float(_percentile(values, 0.95)),
    }


def _aggregate_bytes(values: list[float], *, per_second: bool = False) -> dict[str, float | str] | None:
    if not values:
        return None
    scale = 1024.0 * 1024.0
    stats = _aggregate([float(v) / scale for v in values])
    assert stats is not None
    rounded: dict[str, float | str] = {k: round(v, 4) for k, v in stats.items()}
    rounded["unit"] = "MB/s" if per_second else "MB"
    return rounded


def build_empty_summary(
    *,
    session_id: str,
    label: str,
    metadata: dict[str, Any],
    start_ts: float,
    end_ts: float,
    interval_ms: int,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "label": label,
        "metadata": metadata,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_ms": max(0.0, (end_ts - start_ts) * 1000.0),
        "interval_ms": interval_ms,
        "sample_count": 0,
        "phase_event_count": 0,
        "capabilities": capabilities or {},
        "gpus": [],
        "host": {},
        "phase_summary": {},
        "bottleneck_hypothesis": "No evidence collected.",
        "next_focus": "collect_more_evidence",
        "hints": [],
        "report_text": "No samples or phase events collected.",
    }


def _summarize_gpus(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        for gpu in sample.get("gpus", []):
            grouped[int(gpu["gpu_index"])].append(gpu)

    result = []
    for gpu_index in sorted(grouped):
        rows = grouped[gpu_index]
        result.append(
            {
                "gpu_index": gpu_index,
                "gpu_util_pct": _aggregate([float(r["gpu_util_pct"]) for r in rows if r.get("gpu_util_pct") is not None]),
                "memory_used_mb": _aggregate_bytes([float(r["memory_used_bytes"]) for r in rows if r.get("memory_used_bytes") is not None]),
                "memory_util_pct": _aggregate([float(r["memory_util_pct"]) for r in rows if r.get("memory_util_pct") is not None]),
                "pcie_tx_mb_s": _aggregate_bytes(
                    [float(r["pcie_tx_bytes_s"]) for r in rows if r.get("pcie_tx_bytes_s") is not None],
                    per_second=True,
                ),
                "pcie_rx_mb_s": _aggregate_bytes(
                    [float(r["pcie_rx_bytes_s"]) for r in rows if r.get("pcie_rx_bytes_s") is not None],
                    per_second=True,
                ),
                "power_w": _aggregate([float(r["power_w"]) for r in rows if r.get("power_w") is not None]),
                "temperature_c": _aggregate([float(r["temperature_c"]) for r in rows if r.get("temperature_c") is not None]),
            }
        )
    return result


def _summarize_host(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [sample.get("host", {}) for sample in samples]
    return {
        "cpu_util_pct": _aggregate([float(r["cpu_util_pct"]) for r in rows if r.get("cpu_util_pct") is not None]),
        "memory_util_pct": _aggregate([float(r["memory_util_pct"]) for r in rows if r.get("memory_util_pct") is not None]),
        "memory_used_mb": _aggregate_bytes([float(r["memory_used_bytes"]) for r in rows if r.get("memory_used_bytes") is not None]),
        "disk_read_mb_s": _aggregate_bytes(
            [float(r["disk_read_bytes_s"]) for r in rows if r.get("disk_read_bytes_s") is not None],
            per_second=True,
        ),
        "disk_write_mb_s": _aggregate_bytes(
            [float(r["disk_write_bytes_s"]) for r in rows if r.get("disk_write_bytes_s") is not None],
            per_second=True,
        ),
        "net_rx_mb_s": _aggregate_bytes(
            [float(r["net_rx_bytes_s"]) for r in rows if r.get("net_rx_bytes_s") is not None],
            per_second=True,
        ),
        "net_tx_mb_s": _aggregate_bytes(
            [float(r["net_tx_bytes_s"]) for r in rows if r.get("net_tx_bytes_s") is not None],
            per_second=True,
        ),
    }


def _normalize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for event in events:
        normalized.append(
            {
                "phase": str(event.get("phase", "unknown")),
                "event": str(event.get("event", "")).lower(),
                "step": event.get("step"),
                "ts": float(event.get("ts", 0.0)),
                "metadata": dict(event.get("metadata") or {}),
            }
        )
    return sorted(normalized, key=lambda x: (x["ts"], str(x.get("step"))))


def _summarize_phases(events: list[dict[str, Any]]) -> dict[str, Any]:
    durations: dict[str, list[float]] = defaultdict(list)
    open_events: dict[tuple[Any, str], float] = {}
    for event in _normalize_events(events):
        key = (event.get("step"), event["phase"])
        if event["event"] == "start":
            open_events[key] = event["ts"]
        elif event["event"] == "end" and key in open_events:
            dt = max(0.0, float(event["ts"]) - float(open_events.pop(key)))
            durations[event["phase"]].append(dt)

    summary = {}
    means = {}
    for phase, values in durations.items():
        agg = _aggregate(values)
        if agg is None:
            continue
        rounded_agg = {k: round(v, 6) for k, v in agg.items()}
        summary[phase] = {**rounded_agg, "count": len(values)}
        means[phase] = rounded_agg["avg"]

    total_mean = sum(means.values())
    for phase, agg in summary.items():
        agg["share_pct"] = round(100.0 * agg["avg"] / total_mean, 2) if total_mean > 0 else 0.0
    return summary


def _derive_hints(
    phase_summary: dict[str, Any], gpus: list[dict[str, Any]], host: dict[str, Any]
) -> tuple[list[str], str, str]:
    hints: list[str] = []
    primary = None
    max_share = -1.0
    for phase, agg in phase_summary.items():
        share = float(agg.get("share_pct", 0.0))
        if share > max_share:
            max_share = share
            primary = phase

    gpu_util = None
    pcie_tx = None
    if gpus and gpus[0].get("gpu_util_pct"):
        gpu_util = gpus[0]["gpu_util_pct"]["avg"]
    if gpus and gpus[0].get("pcie_tx_mb_s"):
        pcie_tx = gpus[0]["pcie_tx_mb_s"]["avg"]
    disk_read = host.get("disk_read_mb_s", {}).get("avg") if host.get("disk_read_mb_s") else None
    cpu_util = host.get("cpu_util_pct", {}).get("avg") if host.get("cpu_util_pct") else None

    if primary == "Data":
        hints.append("Data phase dominates measured step time.")
        if gpu_util is not None and gpu_util < 50:
            hints.append("GPU utilization is low while Data dominates; input pipeline is likely the bottleneck.")
        if disk_read is not None and disk_read > 50:
            hints.append("Host disk read throughput is elevated during diagnosis; storage or preprocessing may contribute to stalls.")
    if primary == "H2D":
        hints.append("H2D phase dominates measured step time.")
        if pcie_tx is not None and pcie_tx > 100:
            hints.append("PCIe TX throughput is elevated while H2D dominates; host-to-device transfer is a likely bottleneck.")
    if primary in {"Fwd", "Bwd", "Opt"}:
        hints.append(f"{primary} dominates measured step time.")
        if gpu_util is not None and gpu_util >= 60:
            hints.append("GPU utilization is healthy during compute-heavy phases; workload may be compute-bound.")
        elif gpu_util is not None and gpu_util < 50:
            hints.append("Compute phases dominate but GPU utilization remains modest; check sync points, small batch size, or overlap gaps.")
    if cpu_util is not None and cpu_util > 80:
        hints.append("Host CPU utilization is high; Python preprocessing, dataloader workers, or logging may be limiting throughput.")

    if primary == "Data":
        bottleneck = "Input pipeline / host-side data preparation is the primary bottleneck."
        next_focus = "input_pipeline"
    elif primary == "H2D":
        bottleneck = "Host-to-device transfer is the primary bottleneck."
        next_focus = "transfer_overlap"
    elif primary in {"Fwd", "Bwd", "Opt"}:
        bottleneck = f"{primary} is the dominant measured phase."
        next_focus = "compute_or_sync"
    elif primary:
        bottleneck = f"{primary} is currently the dominant measured phase."
        next_focus = "review_phase_breakdown"
    else:
        bottleneck = "No complete phase timing pairs were collected."
        next_focus = "add_phase_instrumentation"
    return hints, bottleneck, next_focus


def _build_report_text(
    label: str,
    phase_summary: dict[str, Any],
    gpus: list[dict[str, Any]],
    host: dict[str, Any],
    hints: list[str],
    bottleneck: str,
    next_focus: str,
) -> str:
    lines = [f"[Diagnostic Report] label={label}"]
    if phase_summary:
        lines.append("Phase Breakdown:")
        for phase, agg in sorted(phase_summary.items(), key=lambda kv: kv[1]["avg"], reverse=True):
            lines.append(
                f"- {phase}: mean={agg['avg']:.6f}s p95={agg['p95']:.6f}s max={agg['max']:.6f}s share={agg['share_pct']:.2f}% count={agg['count']}"
            )
    if gpus:
        gpu0 = gpus[0]
        lines.append("GPU Telemetry:")
        if gpu0.get("gpu_util_pct"):
            lines.append(f"- GPU util avg={gpu0['gpu_util_pct']['avg']:.2f}% p95={gpu0['gpu_util_pct']['p95']:.2f}%")
        if gpu0.get("pcie_tx_mb_s"):
            lines.append(f"- PCIe TX avg={gpu0['pcie_tx_mb_s']['avg']:.2f} MB/s p95={gpu0['pcie_tx_mb_s']['p95']:.2f} MB/s")
        if gpu0.get("pcie_rx_mb_s"):
            lines.append(f"- PCIe RX avg={gpu0['pcie_rx_mb_s']['avg']:.2f} MB/s p95={gpu0['pcie_rx_mb_s']['p95']:.2f} MB/s")
    if host:
        lines.append("Host Telemetry:")
        if host.get("cpu_util_pct"):
            lines.append(f"- CPU util avg={host['cpu_util_pct']['avg']:.2f}% p95={host['cpu_util_pct']['p95']:.2f}%")
        if host.get("disk_read_mb_s"):
            lines.append(f"- Disk read avg={host['disk_read_mb_s']['avg']:.2f} MB/s")
        if host.get("disk_write_mb_s"):
            lines.append(f"- Disk write avg={host['disk_write_mb_s']['avg']:.2f} MB/s")
    lines.append(f"Primary Bottleneck: {bottleneck}")
    lines.append(f"Next Focus: {next_focus}")
    if hints:
        lines.append("Hints:")
        for hint in hints:
            lines.append(f"- {hint}")
    return "\n".join(lines)


def summarize_session(
    *,
    session_id: str,
    label: str,
    metadata: dict[str, Any],
    start_ts: float,
    end_ts: float,
    interval_ms: int,
    capabilities: dict[str, Any],
    system_samples: list[dict[str, Any]],
    phase_events: list[dict[str, Any]],
) -> dict[str, Any]:
    if not system_samples and not phase_events:
        return build_empty_summary(
            session_id=session_id,
            label=label,
            metadata=metadata,
            start_ts=start_ts,
            end_ts=end_ts,
            interval_ms=interval_ms,
            capabilities=capabilities,
        )

    gpus = _summarize_gpus(system_samples)
    host = _summarize_host(system_samples)
    phase_summary = _summarize_phases(phase_events)
    hints, bottleneck, next_focus = _derive_hints(phase_summary, gpus, host)
    report_text = _build_report_text(label, phase_summary, gpus, host, hints, bottleneck, next_focus)
    return {
        "session_id": session_id,
        "label": label,
        "metadata": metadata,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_ms": max(0.0, (end_ts - start_ts) * 1000.0),
        "interval_ms": interval_ms,
        "sample_count": len(system_samples),
        "phase_event_count": len(phase_events),
        "capabilities": capabilities,
        "gpus": gpus,
        "host": host,
        "phase_summary": phase_summary,
        "bottleneck_hypothesis": bottleneck,
        "next_focus": next_focus,
        "hints": hints,
        "report_text": report_text,
    }


def parse_jsonl_events(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events

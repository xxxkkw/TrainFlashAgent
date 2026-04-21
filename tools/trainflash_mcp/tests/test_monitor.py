from __future__ import annotations

import time

import pytest


class FakeGPUBackend:
    def __init__(self) -> None:
        self.snapshot_calls = 0

    def capabilities(self, gpu_ids=None):
        return {
            "nvml_available": True,
            "pcie_supported": True,
            "nvlink_supported": False,
            "dram_bw_util_supported": False,
        }

    def snapshot(self, gpu_ids=None):
        self.snapshot_calls += 1
        return [
            {
                "gpu_index": 0,
                "gpu_util_pct": 70.0 + self.snapshot_calls,
                "memory_used_bytes": 1024 * 1024 * 100,
                "memory_total_bytes": 1024 * 1024 * 1000,
                "memory_util_pct": 10.0,
                "pcie_tx_bytes_s": 1000.0,
                "pcie_rx_bytes_s": 2000.0,
                "nvlink_tx_bytes_s": None,
                "nvlink_rx_bytes_s": None,
                "dram_bw_util_pct": None,
                "power_w": 250.0,
                "temperature_c": 65.0,
            }
        ]


class FakeHostBackend:
    def capabilities(self):
        return {
            "host_supported": True,
            "cpu_supported": True,
            "disk_io_supported": True,
            "network_io_supported": True,
        }

    def snapshot(self):
        return {
            "cpu_util_pct": 82.0,
            "memory_util_pct": 40.0,
            "memory_used_bytes": 4 * 1024 * 1024 * 1024,
            "disk_read_bytes_s": 80 * 1024 * 1024,
            "disk_write_bytes_s": 10 * 1024 * 1024,
            "net_rx_bytes_s": 1 * 1024 * 1024,
            "net_tx_bytes_s": 2 * 1024 * 1024,
            "process_count": 123,
        }


def test_package_exports_monitor_api():
    from trainflash_mcp import TrainFlashMonitor

    assert TrainFlashMonitor is not None


def test_monitor_start_stop_returns_basic_summary():
    from trainflash_mcp import TrainFlashMonitor

    monitor = TrainFlashMonitor(interval_ms=10)
    session_id = monitor.start("train_step", metadata={"step": 1})
    time.sleep(0.02)
    summary = monitor.stop(session_id)

    assert summary["session_id"] == session_id
    assert summary["label"] == "train_step"
    assert summary["metadata"] == {"step": 1}
    assert summary["interval_ms"] == 50
    assert summary["duration_ms"] >= 0
    assert "capabilities" in summary


def test_monitor_uses_system_samples_and_phase_events_in_summary():
    from trainflash_mcp import TrainFlashMonitor

    monitor = TrainFlashMonitor(interval_ms=50, gpu_backend=FakeGPUBackend(), host_backend=FakeHostBackend())
    session_id = monitor.start("sampled")
    time.sleep(0.12)
    t0 = time.time()
    monitor.record_trainflash_phase_event(session_id, phase="Data", event="start", step=1, ts=t0)
    monitor.record_trainflash_phase_event(session_id, phase="Data", event="end", step=1, ts=t0 + 0.03)
    monitor.record_trainflash_phase_event(session_id, phase="H2D", event="start", step=1, ts=t0 + 0.03)
    monitor.record_trainflash_phase_event(session_id, phase="H2D", event="end", step=1, ts=t0 + 0.05)
    summary = monitor.stop(session_id)

    assert summary["sample_count"] >= 2
    assert summary["phase_event_count"] == 4
    assert summary["gpus"][0]["gpu_index"] == 0
    assert summary["gpus"][0]["gpu_util_pct"]["avg"] >= 71.0
    assert summary["host"]["cpu_util_pct"]["avg"] == 82.0
    assert summary["phase_summary"]["Data"]["avg"] == pytest.approx(0.03)
    assert summary["phase_summary"]["H2D"]["avg"] == pytest.approx(0.02)
    assert "Primary Bottleneck:" in summary["report_text"]


def test_ingest_trainflash_phase_trace_imports_jsonl_events(tmp_path):
    from trainflash_mcp import TrainFlashMonitor

    monitor = TrainFlashMonitor(gpu_backend=FakeGPUBackend(), host_backend=FakeHostBackend())
    session_id = monitor.start("trace")
    trace = tmp_path / "phase_trace.jsonl"
    trace.write_text(
        '{"step": 3, "phase": "Fwd", "event": "start", "ts": 10.0}\n'
        '{"step": 3, "phase": "Fwd", "event": "end", "ts": 10.04}\n',
        encoding="utf-8",
    )
    result = monitor.ingest_trainflash_phase_trace(session_id, trace_path=str(trace))
    summary = monitor.get_trainflash_summary(session_id)

    assert result == {"imported_event_count": 2, "session_id": session_id}
    assert summary["phase_summary"]["Fwd"]["avg"] == pytest.approx(0.04)


def test_monitor_worker_name_uses_trainflash_prefix():
    from trainflash_mcp import TrainFlashMonitor

    monitor = TrainFlashMonitor(interval_ms=50, gpu_backend=FakeGPUBackend(), host_backend=FakeHostBackend())
    session_id = monitor.start("naming")
    worker_name = monitor._get(session_id).worker.name
    monitor.stop(session_id)

    assert worker_name.startswith("trainflash-")


def test_stop_unknown_session_raises_keyerror():
    from trainflash_mcp import TrainFlashMonitor

    with pytest.raises(KeyError):
        TrainFlashMonitor().stop("missing")

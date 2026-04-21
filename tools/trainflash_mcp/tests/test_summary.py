from __future__ import annotations

import json


def test_summary_payload_is_json_serializable():
    from trainflash_mcp import TrainFlashMonitor
    monitor = TrainFlashMonitor()
    session_id = monitor.start("serialize")
    summary = monitor.stop(session_id)
    encoded = json.dumps(summary)
    assert isinstance(encoded, str)
    assert '"label": "serialize"' in encoded


def test_summary_contains_aggregated_metric_shapes():
    from trainflash_mcp.summary import summarize_session
    summary = summarize_session(
        session_id="s1",
        label="train_step",
        metadata={},
        start_ts=0.0,
        end_ts=1.0,
        interval_ms=100,
        capabilities={"phase_event_supported": True},
        system_samples=[
            {
                "ts": 0.1,
                "gpus": [
                    {
                        "gpu_index": 0,
                        "gpu_util_pct": 40.0,
                        "memory_used_bytes": 104857600,
                        "memory_total_bytes": 1048576000,
                        "memory_util_pct": 10.0,
                        "pcie_tx_bytes_s": 1048576.0,
                        "pcie_rx_bytes_s": 2097152.0,
                        "power_w": 200.0,
                        "temperature_c": 60.0,
                    }
                ],
                "host": {
                    "cpu_util_pct": 70.0,
                    "memory_util_pct": 40.0,
                    "memory_used_bytes": 1073741824,
                    "disk_read_bytes_s": 10485760.0,
                    "disk_write_bytes_s": 5242880.0,
                    "net_rx_bytes_s": 1048576.0,
                    "net_tx_bytes_s": 2097152.0,
                },
            },
            {
                "ts": 0.2,
                "gpus": [
                    {
                        "gpu_index": 0,
                        "gpu_util_pct": 60.0,
                        "memory_used_bytes": 209715200,
                        "memory_total_bytes": 1048576000,
                        "memory_util_pct": 20.0,
                        "pcie_tx_bytes_s": 3145728.0,
                        "pcie_rx_bytes_s": 4194304.0,
                        "power_w": 220.0,
                        "temperature_c": 70.0,
                    }
                ],
                "host": {
                    "cpu_util_pct": 90.0,
                    "memory_util_pct": 50.0,
                    "memory_used_bytes": 2147483648,
                    "disk_read_bytes_s": 20971520.0,
                    "disk_write_bytes_s": 10485760.0,
                    "net_rx_bytes_s": 3145728.0,
                    "net_tx_bytes_s": 4194304.0,
                },
            },
        ],
        phase_events=[
            {"step": 0, "phase": "Data", "event": "start", "ts": 0.1},
            {"step": 0, "phase": "Data", "event": "end", "ts": 0.14},
            {"step": 0, "phase": "H2D", "event": "start", "ts": 0.14},
            {"step": 0, "phase": "H2D", "event": "end", "ts": 0.16},
        ],
    )
    gpu0 = summary["gpus"][0]
    assert gpu0["gpu_util_pct"]["avg"] == 50.0
    assert gpu0["memory_used_mb"]["max"] == 200.0
    assert gpu0["pcie_rx_mb_s"]["avg"] == 3.0
    assert summary["host"]["cpu_util_pct"]["avg"] == 80.0
    assert summary["phase_summary"]["Data"]["avg"] == 0.04
    assert summary["phase_summary"]["H2D"]["avg"] == 0.02
    assert summary["next_focus"] == "input_pipeline"

from __future__ import annotations


class FakeGPUBackend:
    def capabilities(self, gpu_ids=None):
        return {
            "nvml_available": True,
            "pcie_supported": True,
            "nvlink_supported": False,
            "dram_bw_util_supported": False,
            "coarse_h2d_d2h_supported": True,
        }

    def snapshot(self, gpu_ids=None):
        return [
            {
                "gpu_index": 0,
                "gpu_name": "Fake GPU",
                "gpu_util_pct": 75.0,
                "memory_used_bytes": 1024,
                "memory_total_bytes": 4096,
                "memory_util_pct": 25.0,
                "pcie_tx_bytes_s": 100.0,
                "pcie_rx_bytes_s": 200.0,
                "nvlink_tx_bytes_s": None,
                "nvlink_rx_bytes_s": None,
                "dram_bw_util_pct": None,
                "power_w": 250.0,
                "temperature_c": 65.0,
                "sm_clock_mhz": 1800.0,
                "mem_clock_mhz": 9000.0,
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
            "cpu_util_pct": 80.0,
            "memory_util_pct": 40.0,
            "memory_used_bytes": 1234,
            "disk_read_bytes_s": 10.0,
            "disk_write_bytes_s": 20.0,
            "net_rx_bytes_s": 30.0,
            "net_tx_bytes_s": 40.0,
            "process_count": 111,
        }


class FakeProfilerBackend:
    def capabilities(self):
        return {
            "nsys_available": False,
            "ncu_available": False,
            "precise_gpu_timeline_supported": False,
            "precise_h2d_d2h_supported": False,
            "recommended_collection_mode": "telemetry_only",
        }


def test_monitor_exposes_live_snapshot_and_combined_capabilities():
    from trainflash_mcp import TrainFlashMonitor

    monitor = TrainFlashMonitor(
        gpu_backend=FakeGPUBackend(),
        host_backend=FakeHostBackend(),
        profiler_backend=FakeProfilerBackend(),
    )

    snapshot = monitor.get_trainflash_system_snapshot()

    assert snapshot["gpus"][0]["gpu_name"] == "Fake GPU"
    assert snapshot["gpus"][0]["sm_clock_mhz"] == 1800.0
    assert snapshot["host"]["process_count"] == 111
    assert snapshot["capabilities"]["coarse_h2d_d2h_supported"] is True
    assert snapshot["capabilities"]["precise_h2d_d2h_supported"] is False
    assert snapshot["collection_mode"] == "telemetry_only"

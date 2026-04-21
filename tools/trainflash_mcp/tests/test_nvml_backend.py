from __future__ import annotations

from types import SimpleNamespace


class FakeHandle:
    def __init__(self, index: int) -> None:
        self.index = index


class FakeNVML:
    NVML_TEMPERATURE_GPU = 0
    NVML_PCIE_UTIL_TX_BYTES = 1
    NVML_PCIE_UTIL_RX_BYTES = 2
    NVML_CLOCK_SM = 10
    NVML_CLOCK_MEM = 11

    def __init__(self) -> None:
        self.initialized = False
        self.handles = [FakeHandle(0), FakeHandle(1)]

    def nvmlInit(self):
        self.initialized = True

    def nvmlShutdown(self):
        self.initialized = False

    def nvmlDeviceGetCount(self):
        return len(self.handles)

    def nvmlDeviceGetHandleByIndex(self, index):
        return self.handles[index]

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=70 + handle.index, memory=20 + handle.index)

    def nvmlDeviceGetMemoryInfo(self, handle):
        used = (handle.index + 1) * 100 * 1024 * 1024
        total = 1000 * 1024 * 1024
        return SimpleNamespace(used=used, total=total)

    def nvmlDeviceGetPowerUsage(self, handle):
        return (250 + handle.index) * 1000

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 65 + handle.index

    def nvmlDeviceGetPcieThroughput(self, handle, counter):
        if counter == self.NVML_PCIE_UTIL_TX_BYTES:
            return 111 + handle.index
        if counter == self.NVML_PCIE_UTIL_RX_BYTES:
            return 222 + handle.index
        raise ValueError(counter)

    def nvmlDeviceGetName(self, handle):
        return f"Fake GPU {handle.index}".encode()

    def nvmlDeviceGetClockInfo(self, handle, clock_type):
        if clock_type == self.NVML_CLOCK_SM:
            return 1800 + handle.index
        if clock_type == self.NVML_CLOCK_MEM:
            return 9000 + handle.index
        raise ValueError(clock_type)


class FakeNVMLNoPcie(FakeNVML):
    def nvmlDeviceGetPcieThroughput(self, handle, counter):
        raise RuntimeError("unsupported")



def test_capabilities_report_false_when_pynvml_missing(monkeypatch):
    import trainflash_mcp.nvml_backend as mod

    monkeypatch.setattr(mod, "pynvml", None)
    backend = mod.NVMLBackend()

    assert backend.capabilities() == {
        "nvml_available": False,
        "pcie_supported": False,
        "nvlink_supported": False,
        "dram_bw_util_supported": False,
        "coarse_h2d_d2h_supported": False,
    }



def test_snapshot_reads_metrics_from_pynvml(monkeypatch):
    import trainflash_mcp.nvml_backend as mod

    fake = FakeNVML()
    monkeypatch.setattr(mod, "pynvml", fake)
    backend = mod.NVMLBackend()

    rows = backend.snapshot()

    assert len(rows) == 2
    assert rows[0]["gpu_index"] == 0
    assert rows[0]["gpu_util_pct"] == 70.0
    assert rows[0]["memory_used_bytes"] == 100 * 1024 * 1024
    assert rows[0]["memory_total_bytes"] == 1000 * 1024 * 1024
    assert rows[0]["memory_util_pct"] == 20.0
    assert rows[0]["pcie_tx_bytes_s"] == 111 * 1024.0
    assert rows[0]["pcie_rx_bytes_s"] == 222 * 1024.0
    assert rows[0]["power_w"] == 250.0
    assert rows[0]["temperature_c"] == 65.0
    assert rows[0]["gpu_name"] == "Fake GPU 0"
    assert rows[0]["sm_clock_mhz"] == 1800.0
    assert rows[0]["mem_clock_mhz"] == 9000.0
    assert rows[0]["nvlink_tx_bytes_s"] is None
    assert rows[0]["dram_bw_util_pct"] is None



def test_snapshot_gracefully_falls_back_when_pcie_unsupported(monkeypatch):
    import trainflash_mcp.nvml_backend as mod

    fake = FakeNVMLNoPcie()
    monkeypatch.setattr(mod, "pynvml", fake)
    backend = mod.NVMLBackend()

    rows = backend.snapshot([0])

    assert len(rows) == 1
    assert rows[0]["pcie_tx_bytes_s"] is None
    assert rows[0]["pcie_rx_bytes_s"] is None



def test_capabilities_detect_pcie_support(monkeypatch):
    import trainflash_mcp.nvml_backend as mod

    fake = FakeNVML()
    monkeypatch.setattr(mod, "pynvml", fake)
    backend = mod.NVMLBackend()

    caps = backend.capabilities([0])

    assert caps["nvml_available"] is True
    assert caps["pcie_supported"] is True
    assert caps["coarse_h2d_d2h_supported"] is True
    assert caps["nvlink_supported"] is False
    assert caps["dram_bw_util_supported"] is False

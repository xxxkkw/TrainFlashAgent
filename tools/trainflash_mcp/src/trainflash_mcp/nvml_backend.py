from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None


@dataclass
class NVMLBackend:
    gpu_ids: list[int] | None = None

    def _resolve_gpu_ids(self, gpu_ids: list[int] | None = None) -> list[int]:
        requested = gpu_ids if gpu_ids is not None else self.gpu_ids
        if requested is not None:
            return [int(x) for x in requested]
        if pynvml is None:
            return []
        return [int(i) for i in range(int(pynvml.nvmlDeviceGetCount()))]

    def _with_init(self) -> bool:
        if pynvml is None:
            return False
        pynvml.nvmlInit()
        return True

    def _safe_pcie_bytes_per_s(self, handle: Any, counter: int) -> float | None:
        try:
            value = pynvml.nvmlDeviceGetPcieThroughput(handle, counter)
            return float(value) * 1024.0
        except Exception:
            return None

    def capabilities(self, gpu_ids: list[int] | None = None) -> dict[str, bool]:
        if pynvml is None:
            return {
                "nvml_available": False,
                "pcie_supported": False,
                "nvlink_supported": False,
                "dram_bw_util_supported": False,
                "coarse_h2d_d2h_supported": False,
            }
        try:
            self._with_init()
            resolved = self._resolve_gpu_ids(gpu_ids)
            pcie_supported = False
            if resolved:
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(resolved[0]))
                pcie_supported = self._safe_pcie_bytes_per_s(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) is not None
            return {
                "nvml_available": True,
                "pcie_supported": bool(pcie_supported),
                "nvlink_supported": False,
                "dram_bw_util_supported": False,
                "coarse_h2d_d2h_supported": bool(pcie_supported),
            }
        except Exception:
            return {
                "nvml_available": False,
                "pcie_supported": False,
                "nvlink_supported": False,
                "dram_bw_util_supported": False,
                "coarse_h2d_d2h_supported": False,
            }
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def snapshot(self, gpu_ids: list[int] | None = None) -> list[dict[str, Any]]:
        if pynvml is None:
            return []
        rows: list[dict[str, Any]] = []
        try:
            self._with_init()
            for gpu_index in self._resolve_gpu_ids(gpu_ids):
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power_w = None
                temperature_c = None
                try:
                    power_w = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                except Exception:
                    pass
                try:
                    temperature_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    pass
                gpu_name = None
                sm_clock_mhz = None
                mem_clock_mhz = None
                try:
                    gpu_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(gpu_name, bytes):
                        gpu_name = gpu_name.decode("utf-8", errors="replace")
                except Exception:
                    pass
                try:
                    sm_clock_mhz = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
                except Exception:
                    pass
                try:
                    mem_clock_mhz = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
                except Exception:
                    pass
                rows.append(
                    {
                        "gpu_index": int(gpu_index),
                        "gpu_name": gpu_name,
                        "gpu_util_pct": float(getattr(util, "gpu", 0.0)),
                        "memory_used_bytes": int(getattr(mem, "used", 0)),
                        "memory_total_bytes": int(getattr(mem, "total", 0)),
                        "memory_util_pct": float(getattr(util, "memory", 0.0)),
                        "pcie_tx_bytes_s": self._safe_pcie_bytes_per_s(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES),
                        "pcie_rx_bytes_s": self._safe_pcie_bytes_per_s(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES),
                        "nvlink_tx_bytes_s": None,
                        "nvlink_rx_bytes_s": None,
                        "dram_bw_util_pct": None,
                        "power_w": power_w,
                        "temperature_c": temperature_c,
                        "sm_clock_mhz": sm_clock_mhz,
                        "mem_clock_mhz": mem_clock_mhz,
                    }
                )
            return rows
        except Exception:
            return []
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

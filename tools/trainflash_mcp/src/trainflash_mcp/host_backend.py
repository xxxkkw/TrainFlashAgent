from __future__ import annotations

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


class HostBackend:
    def __init__(self) -> None:
        self._last_disk = None
        self._last_net = None

    def capabilities(self) -> dict[str, bool]:
        return {
            "host_supported": psutil is not None,
            "cpu_supported": psutil is not None,
            "disk_io_supported": psutil is not None,
            "network_io_supported": psutil is not None,
        }

    def snapshot(self) -> dict:
        if psutil is None:
            return {
                "cpu_util_pct": None,
                "memory_util_pct": None,
                "memory_used_bytes": None,
                "disk_read_bytes_s": None,
                "disk_write_bytes_s": None,
                "net_rx_bytes_s": None,
                "net_tx_bytes_s": None,
                "process_count": None,
            }
        vm = psutil.virtual_memory()
        cpu_pct = float(psutil.cpu_percent(interval=None))
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()
        disk_read = None
        disk_write = None
        net_rx = None
        net_tx = None
        if self._last_disk is not None and disk is not None:
            dt = max(1e-6, disk[8] - self._last_disk[8]) if len(disk) > 8 else 1.0
            disk_read = float(disk.read_bytes - self._last_disk.read_bytes) / dt
            disk_write = float(disk.write_bytes - self._last_disk.write_bytes) / dt
        if self._last_net is not None and net is not None:
            dt = max(1e-6, net[4] - self._last_net[4]) if len(net) > 4 else 1.0
            net_rx = float(net.bytes_recv - self._last_net.bytes_recv) / dt
            net_tx = float(net.bytes_sent - self._last_net.bytes_sent) / dt
        if disk is not None:
            try:
                disk = disk._replace(read_time=getattr(disk, 'read_time', 0), write_time=getattr(disk, 'write_time', 0), busy_time=getattr(disk, 'busy_time', 0))
            except Exception:
                pass
        self._last_disk = disk
        self._last_net = net
        return {
            "cpu_util_pct": cpu_pct,
            "memory_util_pct": float(vm.percent),
            "memory_used_bytes": int(vm.used),
            "disk_read_bytes_s": disk_read,
            "disk_write_bytes_s": disk_write,
            "net_rx_bytes_s": net_rx,
            "net_tx_bytes_s": net_tx,
            "process_count": len(psutil.pids()),
        }

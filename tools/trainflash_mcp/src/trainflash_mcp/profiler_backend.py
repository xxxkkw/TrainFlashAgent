from __future__ import annotations

import shutil


class ProfilerBackend:
    def capabilities(self) -> dict[str, bool | str]:
        nsys_available = shutil.which("nsys") is not None
        ncu_available = shutil.which("ncu") is not None
        iostat_available = shutil.which("iostat") is not None
        pidstat_available = shutil.which("pidstat") is not None
        precise_gpu_timeline_supported = nsys_available or ncu_available
        precise_h2d_d2h_supported = nsys_available
        host_io_cli_supported = iostat_available
        process_io_cli_supported = pidstat_available
        recommended_collection_mode = "telemetry_plus_profiler" if precise_gpu_timeline_supported else "telemetry_only"
        return {
            "nsys_available": nsys_available,
            "ncu_available": ncu_available,
            "iostat_available": iostat_available,
            "pidstat_available": pidstat_available,
            "precise_gpu_timeline_supported": precise_gpu_timeline_supported,
            "precise_h2d_d2h_supported": precise_h2d_d2h_supported,
            "host_io_cli_supported": host_io_cli_supported,
            "process_io_cli_supported": process_io_cli_supported,
            "recommended_collection_mode": recommended_collection_mode,
        }

from __future__ import annotations


def test_profiler_backend_reports_runtime_tool_capabilities(monkeypatch):
    import trainflash_mcp.profiler_backend as mod

    monkeypatch.setattr(mod.shutil, "which", lambda name: {"iostat": "/usr/bin/iostat", "pidstat": "/usr/bin/pidstat"}.get(name))
    backend = mod.ProfilerBackend()

    caps = backend.capabilities()

    assert caps["nsys_available"] is False
    assert caps["ncu_available"] is False
    assert caps["iostat_available"] is True
    assert caps["pidstat_available"] is True
    assert caps["precise_gpu_timeline_supported"] is False
    assert caps["precise_h2d_d2h_supported"] is False
    assert caps["host_io_cli_supported"] is True
    assert caps["process_io_cli_supported"] is True
    assert caps["recommended_collection_mode"] == "telemetry_only"


def test_profiler_backend_prefers_hybrid_mode_when_nsys_available(monkeypatch):
    import trainflash_mcp.profiler_backend as mod

    monkeypatch.setattr(mod.shutil, "which", lambda name: {"nsys": "/opt/nsys", "iostat": "/usr/bin/iostat"}.get(name))
    backend = mod.ProfilerBackend()

    caps = backend.capabilities()

    assert caps["nsys_available"] is True
    assert caps["precise_gpu_timeline_supported"] is True
    assert caps["precise_h2d_d2h_supported"] is True
    assert caps["recommended_collection_mode"] == "telemetry_plus_profiler"

from __future__ import annotations

import importlib
import sys
import types

import pytest


class FakeFastMCP:
    def __init__(self, name: str, instructions: str | None = None, **kwargs) -> None:
        self.name = name
        self.instructions = instructions
        self.kwargs = kwargs
        self.tools: dict[str, object] = {}
        self.ran_transport: str | None = None

    def tool(self, name: str | None = None, description: str | None = None, **kwargs):
        def decorator(fn):
            tool_name = name or fn.__name__
            self.tools[tool_name] = fn
            return fn
        return decorator

    def run(self, transport: str = "stdio") -> None:
        self.ran_transport = transport


class FakeMonitor:
    def __init__(self) -> None:
        self.start_calls = []
        self.stop_calls = []
        self.summary_calls = []
        self.snapshot_calls = 0
        self.phase_calls = []
        self.ingest_calls = []

    def start(self, label: str, metadata: dict | None = None) -> str:
        self.start_calls.append((label, metadata))
        return "sid-1"

    def stop(self, session_id: str) -> dict:
        self.stop_calls.append(session_id)
        return {"session_id": session_id, "state": "stopped"}

    def get_trainflash_summary(self, session_id: str) -> dict:
        self.summary_calls.append(session_id)
        return {"session_id": session_id, "state": "running"}

    def get_trainflash_system_snapshot(self) -> dict:
        self.snapshot_calls += 1
        return {"collection_mode": "telemetry_only", "gpus": [], "host": {}, "capabilities": {}}

    def record_trainflash_phase_event(self, session_id: str, **kwargs) -> dict:
        self.phase_calls.append((session_id, kwargs))
        return {"session_id": session_id, **kwargs}

    def ingest_trainflash_phase_trace(self, session_id: str, **kwargs) -> dict:
        self.ingest_calls.append((session_id, kwargs))
        return {"session_id": session_id, "imported_event_count": 2}

    def capabilities(self) -> dict:
        return {"phase_event_supported": True}


@pytest.fixture()
def mcp_server_module(monkeypatch):
    fake_fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp", types.ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", types.ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp_module)
    sys.modules.pop("trainflash_mcp.mcp_server", None)
    module = importlib.import_module("trainflash_mcp.mcp_server")
    return importlib.reload(module)


def test_create_trainflash_server_registers_expected_tools(mcp_server_module):
    server = mcp_server_module.create_trainflash_server(monitor=FakeMonitor())
    assert server.name == "trainflash_mcp"
    assert set(server.tools) == {
        "start_trainflash_session",
        "record_trainflash_phase_event",
        "ingest_trainflash_phase_trace",
        "get_trainflash_summary",
        "get_trainflash_system_snapshot",
        "stop_trainflash_session",
        "get_trainflash_capabilities",
    }


def test_registered_tools_delegate_to_monitor(mcp_server_module):
    monitor = FakeMonitor()
    server = mcp_server_module.create_trainflash_server(monitor=monitor)
    start_result = server.tools["start_trainflash_session"]("train_step", {"step": 1})
    phase_result = server.tools["record_trainflash_phase_event"]("sid-1", "Data", "start", 1, 10.0, {"x": 1})
    ingest_result = server.tools["ingest_trainflash_phase_trace"]("sid-1", "/tmp/x.jsonl", None)
    summary_result = server.tools["get_trainflash_summary"]("sid-1")
    snapshot_result = server.tools["get_trainflash_system_snapshot"]()
    stop_result = server.tools["stop_trainflash_session"]("sid-1")
    caps_result = server.tools["get_trainflash_capabilities"]()
    assert start_result == {"session_id": "sid-1"}
    assert phase_result["phase"] == "Data"
    assert ingest_result == {"session_id": "sid-1", "imported_event_count": 2}
    assert summary_result == {"session_id": "sid-1", "state": "running"}
    assert snapshot_result == {"collection_mode": "telemetry_only", "gpus": [], "host": {}, "capabilities": {}}
    assert stop_result == {"session_id": "sid-1", "state": "stopped"}
    assert caps_result == {"phase_event_supported": True}
    assert monitor.snapshot_calls == 1


def test_run_trainflash_stdio_builds_server_and_runs_stdio_transport(mcp_server_module):
    server = mcp_server_module.run_trainflash_stdio(monitor=FakeMonitor())
    assert isinstance(server, FakeFastMCP)
    assert server.ran_transport == "stdio"


def test_create_trainflash_server_raises_clear_error_without_mcp(monkeypatch):
    sys.modules.pop("trainflash_mcp.mcp_server", None)
    monkeypatch.delitem(sys.modules, "mcp.server.fastmcp", raising=False)
    monkeypatch.delitem(sys.modules, "mcp.server", raising=False)
    monkeypatch.delitem(sys.modules, "mcp", raising=False)
    module = importlib.import_module("trainflash_mcp.mcp_server")
    module = importlib.reload(module)
    with pytest.raises(RuntimeError, match=r"Install trainflash-mcp\[mcp\] or mcp"):
        module.create_trainflash_server(monitor=FakeMonitor())

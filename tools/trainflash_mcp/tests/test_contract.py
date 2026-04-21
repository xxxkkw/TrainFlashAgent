from pathlib import Path

import trainflash_mcp.mcp_server as mcp_server
from trainflash_mcp import create_trainflash_server


class FakeFastMCP:
    def __init__(self, name, instructions=None, **kwargs):
        self.name = name
        self.instructions = instructions
        self.kwargs = kwargs
        self.tools = {}

    def tool(self, name=None, description=None, **kwargs):
        def decorator(fn):
            self.tools[name or fn.__name__] = fn
            return fn

        return decorator


EXPECTED_TOOLS = {
    "get_trainflash_capabilities",
    "get_trainflash_summary",
    "get_trainflash_system_snapshot",
    "ingest_trainflash_phase_trace",
    "record_trainflash_phase_event",
    "start_trainflash_session",
    "stop_trainflash_session",
}


def test_server_exposes_expected_tool_contract(monkeypatch):
    monkeypatch.setattr(mcp_server, "FastMCP", FakeFastMCP)

    server = create_trainflash_server()

    assert set(server.tools) == EXPECTED_TOOLS


def test_readme_documents_hermes_host_integration():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")

    assert "Hermes config example" in text
    assert "mcp_servers:" in text
    assert "trainflash:" in text
    assert 'command: "python"' in text
    assert "python -m trainflash_mcp" in text

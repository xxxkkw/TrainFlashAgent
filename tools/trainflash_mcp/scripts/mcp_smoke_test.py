from __future__ import annotations

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trainflash_mcp import create_trainflash_server


class FakeFastMCP:
    def __init__(self, name: str, instructions: str | None = None, **kwargs) -> None:
        self.name = name
        self.instructions = instructions
        self.kwargs = kwargs
        self.tools: dict[str, object] = {}

    def tool(self, name: str | None = None, description: str | None = None, **kwargs):
        def decorator(fn):
            self.tools[name or fn.__name__] = fn
            return fn

        return decorator

    def run(self, transport: str = "stdio") -> None:
        raise RuntimeError("Smoke test should not call run()")


def main() -> int:
    fake_fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp_module.FastMCP = FakeFastMCP
    sys.modules.setdefault("mcp", types.ModuleType("mcp"))
    sys.modules.setdefault("mcp.server", types.ModuleType("mcp.server"))
    sys.modules["mcp.server.fastmcp"] = fake_fastmcp_module

    import trainflash_mcp.mcp_server as mcp_server

    mcp_server.FastMCP = FakeFastMCP
    server = create_trainflash_server()
    payload = {
        "server_name": server.name,
        "tool_names": sorted(server.tools),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

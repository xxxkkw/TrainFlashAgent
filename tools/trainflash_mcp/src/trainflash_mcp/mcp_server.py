from __future__ import annotations

from typing import Any

from .monitor import TrainFlashMonitor

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None

SERVER_NAME = "trainflash_mcp"
SERVER_INSTRUCTIONS = (
    "Lightweight TrainFlash MCP server exposing session-based system telemetry, "
    "phase timing ingestion, and summary/report generation for TrainFlashAgent."
)


def _require_fastmcp() -> type:
    if FastMCP is None:
        raise RuntimeError("MCP dependency is not installed. Install trainflash-mcp[mcp] or mcp.")
    return FastMCP


def create_trainflash_server(monitor: TrainFlashMonitor | None = None, *, interval_ms: int = 100, gpu_ids: list[int] | None = None) -> Any:
    fastmcp_cls = _require_fastmcp()
    active_monitor = monitor or TrainFlashMonitor(interval_ms=interval_ms, gpu_ids=gpu_ids)
    server = fastmcp_cls(name=SERVER_NAME, instructions=SERVER_INSTRUCTIONS)

    @server.tool(name="start_trainflash_session", description="Start a TrainFlash session and begin background telemetry sampling.")
    def start_trainflash_session(label: str, metadata: dict[str, Any] | None = None) -> dict[str, str]:
        return {"session_id": active_monitor.start(label=label, metadata=metadata)}

    @server.tool(name="record_trainflash_phase_event", description="Record a single phase timing event for Data/H2D/Fwd/Bwd/Opt/Eval/Ckpt/Log.")
    def record_trainflash_phase_event(session_id: str, phase: str, event: str, step: int | None = None, ts: float | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return active_monitor.record_trainflash_phase_event(session_id, phase=phase, event=event, step=step, ts=ts, metadata=metadata)

    @server.tool(name="ingest_trainflash_phase_trace", description="Import a JSONL phase trace file or inline event list into an existing TrainFlash session.")
    def ingest_trainflash_phase_trace(session_id: str, trace_path: str | None = None, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return active_monitor.ingest_trainflash_phase_trace(session_id, trace_path=trace_path, events=events)

    @server.tool(name="get_trainflash_summary", description="Get the current aggregated diagnosis summary and TrainFlash report.")
    def get_trainflash_summary(session_id: str) -> dict[str, Any]:
        return active_monitor.get_trainflash_summary(session_id)

    @server.tool(name="get_trainflash_system_snapshot", description="Return a live telemetry snapshot for GPU core util, PCIe proxy H2D/D2H throughput, host IO, and capability flags.")
    def get_trainflash_system_snapshot() -> dict[str, Any]:
        return active_monitor.get_trainflash_system_snapshot()

    @server.tool(name="stop_trainflash_session", description="Stop a TrainFlash session and return the final aggregated summary/report.")
    def stop_trainflash_session(session_id: str) -> dict[str, Any]:
        return active_monitor.stop(session_id)

    @server.tool(name="get_trainflash_capabilities", description="Return which telemetry and phase-ingestion capabilities are available on this machine.")
    def get_trainflash_capabilities() -> dict[str, Any]:
        return active_monitor.capabilities()

    return server


def run_trainflash_stdio(monitor: TrainFlashMonitor | None = None, *, interval_ms: int = 100, gpu_ids: list[int] | None = None) -> Any:
    server = create_trainflash_server(monitor=monitor, interval_ms=interval_ms, gpu_ids=gpu_ids)
    server.run(transport="stdio")
    return server

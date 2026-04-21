from __future__ import annotations

import time
import uuid
from pathlib import Path
from threading import Event, Thread
from typing import Any

from .host_backend import HostBackend
from .nvml_backend import NVMLBackend
from .profiler_backend import ProfilerBackend
from .session import TrainFlashSession
from .summary import parse_jsonl_events, summarize_session


class TrainFlashMonitor:
    def __init__(self, interval_ms: int = 100, gpu_ids: list[int] | None = None, gpu_backend: Any = None, host_backend: Any = None, profiler_backend: Any = None) -> None:
        self.interval_ms = max(50, int(interval_ms))
        self.gpu_ids = gpu_ids
        self.gpu_backend = gpu_backend or NVMLBackend(gpu_ids=gpu_ids)
        self.host_backend = host_backend or HostBackend()
        self.profiler_backend = profiler_backend or ProfilerBackend()
        self._sessions: dict[str, TrainFlashSession] = {}

    def capabilities(self) -> dict[str, Any]:
        profiler_caps = self.profiler_backend.capabilities()
        return {
            **self.gpu_backend.capabilities(self.gpu_ids),
            **self.host_backend.capabilities(),
            **profiler_caps,
            "phase_event_supported": True,
            "jsonl_trace_supported": True,
        }

    def get_trainflash_system_snapshot(self) -> dict[str, Any]:
        capabilities = self.capabilities()
        return {
            "ts": time.time(),
            "gpus": self.gpu_backend.snapshot(self.gpu_ids),
            "host": self.host_backend.snapshot(),
            "capabilities": capabilities,
            "collection_mode": capabilities.get("recommended_collection_mode", "telemetry_only"),
        }

    def start(self, label: str, metadata: dict[str, Any] | None = None) -> str:
        session_id = str(uuid.uuid4())
        stop_event = Event()
        session = TrainFlashSession(
            session_id=session_id,
            label=str(label),
            metadata=dict(metadata or {}),
            start_ts=time.time(),
            capabilities=self.capabilities(),
            stop_event=stop_event,
        )
        worker = Thread(target=self._sample_loop, args=(session,), daemon=True, name=f"trainflash-{session_id[:8]}")
        session.worker = worker
        self._sessions[session_id] = session
        worker.start()
        return session_id

    def _sample_loop(self, session: TrainFlashSession) -> None:
        assert session.stop_event is not None
        while not session.stop_event.is_set():
            session.system_samples.append(
                {
                    "ts": time.time(),
                    "gpus": self.gpu_backend.snapshot(self.gpu_ids),
                    "host": self.host_backend.snapshot(),
                }
            )
            session.stop_event.wait(self.interval_ms / 1000.0)

    def _get(self, session_id: str) -> TrainFlashSession:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._sessions[session_id]

    def record_trainflash_phase_event(self, session_id: str, *, phase: str, event: str, step: int | None = None, ts: float | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        session = self._get(session_id)
        payload = {
            "phase": str(phase),
            "event": str(event).lower(),
            "step": step,
            "ts": float(time.time() if ts is None else ts),
            "metadata": dict(metadata or {}),
        }
        session.phase_events.append(payload)
        return payload

    def ingest_trainflash_phase_trace(self, session_id: str, *, trace_path: str | None = None, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        session = self._get(session_id)
        imported: list[dict[str, Any]] = []
        if trace_path:
            imported.extend(parse_jsonl_events(Path(trace_path).read_text(encoding='utf-8')))
        if events:
            imported.extend(events)
        for event in imported:
            self.record_trainflash_phase_event(
                session_id,
                phase=event["phase"],
                event=event["event"],
                step=event.get("step"),
                ts=event.get("ts"),
                metadata=event.get("metadata"),
            )
        return {"imported_event_count": len(imported), "session_id": session_id}

    def get_trainflash_summary(self, session_id: str) -> dict[str, Any]:
        session = self._get(session_id)
        if session.summary_payload is not None:
            return session.summary_payload
        return summarize_session(
            session_id=session.session_id,
            label=session.label,
            metadata=session.metadata,
            start_ts=session.start_ts,
            end_ts=time.time(),
            interval_ms=self.interval_ms,
            capabilities=session.capabilities,
            system_samples=session.system_samples,
            phase_events=session.phase_events,
        )

    def stop(self, session_id: str) -> dict[str, Any]:
        session = self._get(session_id)
        if session.summary_payload is not None:
            return session.summary_payload
        if session.stop_event is not None:
            session.stop_event.set()
        if session.worker is not None:
            session.worker.join(timeout=max(1.0, self.interval_ms / 1000.0 * 2.0))
        session.end_ts = time.time()
        session.stopped = True
        session.summary_payload = summarize_session(
            session_id=session.session_id,
            label=session.label,
            metadata=session.metadata,
            start_ts=session.start_ts,
            end_ts=session.end_ts,
            interval_ms=self.interval_ms,
            capabilities=session.capabilities,
            system_samples=session.system_samples,
            phase_events=session.phase_events,
        )
        return session.summary_payload

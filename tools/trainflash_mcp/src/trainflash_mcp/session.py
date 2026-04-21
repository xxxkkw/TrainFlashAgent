from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any


@dataclass
class TrainFlashSession:
    session_id: str
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_ts: float = 0.0
    end_ts: float | None = None
    stopped: bool = False
    summary_payload: dict[str, Any] | None = None
    system_samples: list[dict[str, Any]] = field(default_factory=list)
    phase_events: list[dict[str, Any]] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)
    stop_event: Event | None = None
    worker: Thread | None = None

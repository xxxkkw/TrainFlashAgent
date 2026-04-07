from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import enum

class Phase(enum.Enum):
    MACRO_DIAGNOSTICS = 1
    ENGINEERING_TUNING = 2
    MICRO_DIAGNOSTICS = 3
    MERGED = 4

@dataclass
class TuningEntry:
    modification: str
    expected_gain: str
    measured_gain: Optional[float] = None
    status: str = "Pending"  # Pending -> Verified -> Merged
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class TuningLog:
    """
    Structured audit log for all optimization attempts.
    """
    def __init__(self):
        self.entries: List[TuningEntry] = []

    def add_entry(self, modification: str, expected_gain: str):
        entry = TuningEntry(modification=modification, expected_gain=expected_gain)
        self.entries.append(entry)
        return len(self.entries) - 1

    def update_gain(self, index: int, gain: float, status: str = "Verified"):
        self.entries[index].measured_gain = gain
        self.entries[index].status = status

    def get_report(self) -> List[Dict[str, Any]]:
        return [vars(e) for e in self.entries]

class ProgressionGate:
    """
    Enforces the Top-Down Methodology:
    Ensures a phase is plateaued before allowing progression to the next.
    """
    def __init__(self, plateau_threshold: float = 0.02, plateau_count: int = 3):
        self.plateau_threshold = plateau_threshold
        self.plateau_count = plateau_count
        self.recent_gains: List[float] = []

    def record_gain(self, gain: float):
        self.recent_gains.append(gain)
        if len(self.recent_gains) > self.plateau_count:
            self.recent_gains.pop(0)

    def can_progress(self) -> bool:
        if len(self.recent_gains) < self.plateau_count:
            return False

        # If all recent gains are below threshold, we have plateaued
        return all(abs(g) < self.plateau_threshold for g in self.recent_gains)

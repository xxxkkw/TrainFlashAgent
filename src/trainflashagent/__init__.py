"""
TrainFlashAgent: AI-driven Top-Down Training Performance Optimizer

An Agent-native Skill-based architecture for DL training optimization.
Follows Top-Down Methodology: Macro-Diagnostics → Engineering Tuning → Micro-Diagnostics
"""
from .sandbox import SandboxManager
from .diagnostics import TimerInjector, collect_timer_logs, analyze_time_variance, run_detailed_profiler
from .interventions import EngineeringIntervention
from .verification import VerificationSuite
from .governance import TuningLog, ProgressionGate, Phase
from .manager import TuningManager

__all__ = [
    # Sandbox
    "SandboxManager",
    # Diagnostics
    "TimerInjector",
    "collect_timer_logs",
    "analyze_time_variance",
    "run_detailed_profiler",
    # Interventions
    "EngineeringIntervention",
    # Verification
    "VerificationSuite",
    # Governance
    "TuningLog",
    "ProgressionGate",
    "Phase",
    # Orchestrator
    "TuningManager",
]

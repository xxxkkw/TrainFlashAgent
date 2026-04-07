"""
TrainFlashAgent MCP Server

This server exposes a set of independent Skills that an LLM can use to
perform top-down performance optimization on deep learning training code.

The LLM is in control - it decides when to diagnose, what to optimize,
and when to merge changes back.
"""

import os
import sys
import json
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from trainflashagent.sandbox import SandboxManager
from trainflashagent.diagnostics import TimerInjector, collect_timer_logs, analyze_time_variance, run_detailed_profiler
from trainflashagent.interventions import EngineeringIntervention
from trainflashagent.verification import VerificationSuite

# Initialize MCP server
mcp = FastMCP("TrainFlashAgent")

# Configuration
PROJECT_ROOT = os.getenv("TFA_PROJECT_ROOT", "/data/xxxkw")
SANDBOX_ROOT = os.getenv("TFA_SANDBOX_ROOT", "/tmp/trainflashagent_sandboxes")

# Global tool instances (stateless, LLM controls the flow)
sandbox_manager = SandboxManager(snapshot_base=os.path.join(SANDBOX_ROOT, "snapshots"))
timer_injector = TimerInjector()
intervention_tool = EngineeringIntervention()
verification_suite = VerificationSuite()

# State tracking (minimal - just the active sandbox path)
_active_sandbox = None


# ==================== SANDBOX SKILLS ====================

@mcp.tool()
def create_sandbox(project_root: str, sandbox_name: str = "active") -> Dict[str, Any]:
    """
    Create an isolated sandbox by cloning the target project.

    This is the FIRST step before any optimization work.
    All modifications happen in this sandbox, not the original project.

    Args:
        project_root: Absolute path to the training project to optimize.
        sandbox_name: Name for this sandbox instance.

    Returns:
        {"sandbox_path": str, "status": "created"}
    """
    global _active_sandbox
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    sandbox_manager.sandbox_clone(project_root, sandbox_path)
    _active_sandbox = sandbox_path
    return {"sandbox_path": sandbox_path, "status": "created", "message": f"Sandbox created from {project_root}"}


@mcp.tool()
def create_snapshot(sandbox_name: str = "active", label: str = None) -> Dict[str, Any]:
    """
    Create a snapshot of the current sandbox state before making changes.

    Use this before applying any intervention to enable rollback.

    Args:
        sandbox_name: Name of the sandbox to snapshot.
        label: Optional label for this snapshot (auto-generated if not provided).

    Returns:
        {"snapshot_path": str, "label": str}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    snapshot_path = sandbox_manager.sandbox_snapshot(sandbox_path, label)
    return {"snapshot_path": snapshot_path, "label": os.path.basename(snapshot_path)}


@mcp.tool()
def rollback_to_snapshot(sandbox_name: str, snapshot_label: str) -> Dict[str, Any]:
    """
    Rollback the sandbox to a previous snapshot.

    Use this when an intervention fails verification.

    Args:
        sandbox_name: Name of the sandbox to rollback.
        snapshot_label: Label of the snapshot to restore.

    Returns:
        {"status": "rolled_back", "sandbox_path": str}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    sandbox_manager.sandbox_rollback(sandbox_path, snapshot_label)
    return {"status": "rolled_back", "sandbox_path": sandbox_path, "snapshot_label": snapshot_label}


@mcp.tool()
def merge_to_main(sandbox_name: str, project_root: str) -> Dict[str, Any]:
    """
    Merge verified changes from sandbox back to the original project.

    This is the FINAL step - only call after all optimizations are verified.

    Args:
        sandbox_name: Name of the sandbox containing verified changes.
        project_root: Original project path to merge into.

    Returns:
        {"status": "merged", "files_modified": int}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    sandbox_manager.merge_to_main(sandbox_path, project_root)
    return {"status": "merged", "sandbox_path": sandbox_path, "project_root": project_root}


# ==================== DIAGNOSTIC SKILLS ====================

@mcp.tool()
def inject_timer(file_path: str, target_function: str, label: str, sandbox_name: str = "active") -> Dict[str, Any]:
    """
    Inject manual timing code into a specific function using AST.

    This is PHASE 1 (Macro-Diagnostics) - use this BEFORE any profilers.
    The timer will log output in format: [TRAINOPT_TIMER] {label}: {time}

    Args:
        file_path: Relative path to the Python file from sandbox root.
        target_function: Name of the function to wrap with timing.
        label: Label for this timer (e.g., "DataLoader", "ForwardPass").
        sandbox_name: Name of the sandbox containing the file.

    Returns:
        {"status": "injected", "file_path": str, "function": str}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    full_path = os.path.join(sandbox_path, file_path)

    with open(full_path, 'r') as f:
        code = f.read()

    modified_code = timer_injector.inject_manual_timer(code, target_function, label)

    with open(full_path, 'w') as f:
        f.write(modified_code)

    return {"status": "injected", "file_path": file_path, "function": target_function, "label": label}


@mcp.tool()
def collect_logs(script_path: str, steps: int = 50, sandbox_name: str = "active") -> Dict[str, Any]:
    """
    Run the training script and collect timer logs.

    Executes the script and parses [TRAINOPT_TIMER] outputs.

    Args:
        script_path: Relative path to the training script.
        steps: Number of steps to run (default 50, can be higher for better stats).
        sandbox_name: Name of the sandbox containing the script.

    Returns:
        {"logs": [{"label": str, "time": float}, ...]}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    full_path = os.path.join(sandbox_path, script_path)

    logs = collect_timer_logs(full_path, steps)

    return {"status": "collected", "logs": logs, "count": len(logs)}


@mcp.tool()
def analyze_variance(logs: list) -> Dict[str, Any]:
    """
    Analyze timer logs to identify bottlenecks and long-tail samples.

    Pass the logs from collect_logs() to get statistics.

    Args:
        logs: List of {"label": str, "time": float} from collect_logs.

    Returns:
        {
            "label": {
                "mean": float, "std_dev": float, "max": float, "min": float,
                "count": int, "is_long_tail": bool
            },
            ...
        }
    """
    report = analyze_time_variance(logs)
    return {"status": "analyzed", "report": report}


@mcp.tool()
def run_profiler(script_path: str, steps: int = 10, sandbox_name: str = "active") -> Dict[str, Any]:
    """
    Run detailed torch.profiler for micro-level (kernel) analysis.

    This is PHASE 3 - only use after Phase 2 (engineering tuning) has plateaued.

    Args:
        script_path: Relative path to the training script.
        steps: Number of steps to profile.
        sandbox_name: Name of the sandbox.

    Returns:
        {"status": str, "trace_path": str, "summary": str}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    full_path = os.path.join(sandbox_path, script_path)
    output_dir = os.path.join(sandbox_path, "profiler_output")

    result = run_detailed_profiler(full_path, steps, output_dir)

    return {
        "status": result["status"],
        "trace_path": result.get("trace_path", ""),
        "summary": result.get("output", "")[-1000:] if result.get("output") else "",
        "message": result.get("message", "")
    }


# ==================== INTERVENTION SKILLS ====================

@mcp.tool()
def optimize_data_pipeline(file_path: str, strategy: str = "balanced", sandbox_name: str = "active") -> Dict[str, Any]:
    """
    Optimize PyTorch DataLoader settings to resolve IO bottlenecks.

    This is PHASE 2 (Engineering Tuning).

    Args:
        file_path: Relative path to the file containing DataLoader.
        strategy: "aggressive" (high workers), "balanced" (moderate), or "conservative".
        sandbox_name: Name of the sandbox.

    Returns:
        {"status": "applied" | "no_change", "file_path": str}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    full_path = os.path.join(sandbox_path, file_path)

    success = intervention_tool.optimize_data_pipeline(full_path, strategy)

    return {"status": "applied" if success else "no_change", "file_path": file_path, "strategy": strategy}


@mcp.tool()
def tune_resources(file_path: str, params: Dict[str, int], sandbox_name: str = "active") -> Dict[str, Any]:
    """
    Tune resource-related hyperparameters like batch_size, gradient_accumulation_steps.

    Args:
        file_path: Relative path to the config or training file.
        params: Dict of {"param_name": new_value}, e.g., {"batch_size": 64, "gradient_accumulation_steps": 4}
        sandbox_name: Name of the sandbox.

    Returns:
        {"status": "applied" | "no_change", "file_path": str, "params": dict}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    full_path = os.path.join(sandbox_path, file_path)

    success = intervention_tool.tune_resource_config(full_path, params)

    return {"status": "applied" if success else "no_change", "file_path": file_path, "params": params}


@mcp.tool()
def edit_code(file_path: str, search_pattern: str, replacement: str, sandbox_name: str = "active") -> Dict[str, Any]:
    """
    General purpose code modification using regex search and replace.

    Use this for custom engineering fixes not covered by other skills.

    Args:
        file_path: Relative path to the file to modify.
        search_pattern: Regex pattern to search for.
        replacement: Replacement string (can use \\1, \\2 for backreferences).
        sandbox_name: Name of the sandbox.

    Returns:
        {"status": "applied" | "no_match", "file_path": str}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    full_path = os.path.join(sandbox_path, file_path)

    success = intervention_tool.edit_code(full_path, search_pattern, replacement)

    return {"status": "applied" if success else "no_match", "file_path": file_path}


# ==================== VERIFICATION SKILLS ====================

@mcp.tool()
def benchmark_throughput(script_path: str, steps: int = 100, sandbox_name: str = "active") -> Dict[str, Any]:
    """
    Measure training throughput (samples/sec or tokens/sec).

    Run this before and after interventions to measure gain.

    Args:
        script_path: Relative path to the training script.
        steps: Number of steps to benchmark (default 100).
        sandbox_name: Name of the sandbox.

    Returns:
        {"status": str, "avg_throughput": float, "max_throughput": float, "min_throughput": float}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    full_path = os.path.join(sandbox_path, script_path)

    result = verification_suite.benchmark_throughput(full_path, steps)

    return result


@mcp.tool()
def verify_fidelity(baseline_losses: list, optimized_losses: list, tolerance: float = 0.001) -> Dict[str, Any]:
    """
    Verify that optimization preserves model fidelity by comparing loss curves.

    Args:
        baseline_losses: List of loss values from original run.
        optimized_losses: List of loss values from optimized run.
        tolerance: Maximum allowed mean difference (default 0.001).

    Returns:
        {"is_preserved": bool, "mean_diff": float, "max_diff": float}
    """
    result = verification_suite.verify_model_fidelity(baseline_losses, optimized_losses, tolerance)
    return result


@mcp.tool()
def read_file(file_path: str, sandbox_name: str = "active") -> Dict[str, Any]:
    """
    Read a file from the sandbox to inspect current code.

    Args:
        file_path: Relative path to the file.
        sandbox_name: Name of the sandbox.

    Returns:
        {"content": str, "file_path": str}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)
    full_path = os.path.join(sandbox_path, file_path)

    with open(full_path, 'r') as f:
        content = f.read()

    return {"content": content, "file_path": file_path}


@mcp.tool()
def list_files(sandbox_name: str = "active") -> Dict[str, Any]:
    """
    List all Python files in the sandbox.

    Args:
        sandbox_name: Name of the sandbox.

    Returns:
        {"files": [str, ...]}
    """
    sandbox_path = os.path.join(SANDBOX_ROOT, sandbox_name)

    files = []
    for root, _, filenames in os.walk(sandbox_path):
        for filename in filenames:
            if filename.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, filename), sandbox_path)
                files.append(rel_path)

    return {"files": files}


if __name__ == "__main__":
    mcp.run()

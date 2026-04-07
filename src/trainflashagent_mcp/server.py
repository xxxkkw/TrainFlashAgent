import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from trainflashagent.manager import TuningManager
from trainflashagent.diagnostics import run_detailed_profiler

# Initialize FastMCP server
mcp = FastMCP("TrainFlashAgent")

# Global manager instance
# In a production setup, these paths would be passed via environment variables or config
PROJECT_ROOT = os.getenv("TFA_PROJECT_ROOT", "/data/xxxkw")
SANDBOX_ROOT = os.getenv("TFA_SANDBOX_ROOT", "/tmp/trainflashagent_sandboxes")

manager = TuningManager(project_root=PROJECT_ROOT, sandbox_root=SANDBOX_ROOT)

@mcp.tool()
def setup_sandbox() -> str:
    """
    Initializes the isolated sandbox environment by cloning the target project.
    This must be called before any other optimization tools.
    """
    try:
        path = manager.setup_environment()
        return f"Sandbox initialized successfully at: {path}"
    except Exception as e:
        return f"Error initializing sandbox: {str(e)}"

@mcp.tool()
def macro_diagnostic(target_file: str, target_func: str, label: str) -> str:
    """
    Injects manual timers into the specified function to identify macro-level bottlenecks.

    Args:
        target_file: Relative path to the python file in the project.
        target_func: Name of the function to profile.
        label: A descriptive label for the timer (e.g., 'DataLoader_Loop').
    """
    try:
        report = manager.run_macro_diagnostic(target_file, target_func, label)
        return f"Macro Diagnostic Report for {label}:\n{report}"
    except Exception as e:
        return f"Error during macro diagnostic: {str(e)}"

@mcp.tool()
def run_micro_profiler(output_dir: str = "./profiler_output") -> str:
    """
    Runs a detailed micro-level profiler (torch.profiler) for kernel-level analysis.
    This is Phase 3 - only use after Phase 2 (engineering tuning) has plateaued.

    Args:
        output_dir: Directory to save profiler traces (Chrome trace JSON).
    """
    try:
        result = run_detailed_profiler(
            sandbox_path=os.path.join(manager.active_sandbox, "train.py"),
            output_dir=output_dir
        )
        if result["status"] == "success":
            return f"Profiler completed successfully.\nTrace saved to: {result['trace_path']}\n\nSummary:\n{result['output'][-500:]}"
        else:
            return f"Profiler failed: {result['message']}"
    except Exception as e:
        return f"Error running profiler: {str(e)}"


@mcp.tool()
def edit_code(file_rel_path: str, search_pattern: str, replacement: str) -> str:
    """
    General purpose code modification using regex search and replace.

    Args:
        file_rel_path: Relative path to the file to modify.
        search_pattern: Regex pattern to search for.
        replacement: Replacement string (can use \\1, \\2 for backreferences).
    """
    try:
        full_path = os.path.join(manager.active_sandbox, file_rel_path)
        success = manager.interventions.edit_code(full_path, search_pattern, replacement)
        if success:
            return f"Code modification applied successfully to {file_rel_path}"
        else:
            return f"No match found for pattern in {file_rel_path}. No changes made."
    except Exception as e:
        return f"Error editing code: {str(e)}"


@mcp.tool()
def apply_intervention(file_rel_path: str, intervention_type: str, params: dict = None) -> str:
    """
    Applies an engineering fix and immediately verifies its impact on throughput.

    Args:
        file_rel_path: Relative path to the file to modify.
        intervention_type: One of 'data_pipeline', 'resource_config', or 'equivalent_sub'.
        params: Configuration parameters for the intervention.
    """
    try:
        result = manager.apply_and_verify_intervention(file_rel_path, intervention_type, params)
        if result["status"] == "success":
            return f"Intervention applied successfully. Measured Gain (Throughput): {result['gain']}"
        elif result["status"] == "no_change":
            return "Intervention applied but no change in code detected."
        else:
            return f"Intervention failed: {result.get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error applying intervention: {str(e)}"

@mcp.tool()
def check_phase_progression() -> str:
    """
    Checks if the current optimization phase has plateaued and if the agent can progress to the next phase.
    Returns whether progression is permitted.
    """
    try:
        can_progress = manager.check_progression()
        phase = manager.current_phase.name
        status = "PERMITTED" if can_progress else "DENIED (Still improving)"
        return f"Current Phase: {phase}\nProgression to next phase: {status}"
    except Exception as e:
        return f"Error checking progression: {str(e)}"

@mcp.tool()
def get_gain_report() -> str:
    """
    Generates a report of pending optimization gains for approval workflow.
    This is part of Section 4.3: Reporting & Approval Workflow.
    """
    try:
        return manager.get_gain_report()
    except Exception as e:
        return f"Error generating gain report: {str(e)}"


@mcp.tool()
def approve_pending_gains() -> str:
    """
    Commits pending gains to baseline after user approval.
    This finalizes the optimization and updates the baseline throughput.
    """
    try:
        manager.approve_pending_gains()
        return "Pending gains have been approved and committed to baseline."
    except Exception as e:
        return f"Error approving gains: {str(e)}"


@mcp.tool()
def get_tuning_log() -> str:
    """
    Retrieves the full structured audit log of all optimization attempts and their results.
    """
    try:
        logs = manager.log.get_report()
        if not logs:
            return "No tuning entries recorded yet."

        report = "Tuning Audit Log:\n"
        for i, entry in enumerate(logs):
            report += f"[{i}] {entry['timestamp']} | {entry['modification']} | Gain: {entry['measured_gain']} | Status: {entry['status']}\n"
        return report
    except Exception as e:
        return f"Error retrieving logs: {str(e)}"

@mcp.tool()
def merge_to_main() -> str:
    """
    Merges all verified changes from the sandbox back to the original project root.
    This is the final step of the optimization workflow.
    """
    try:
        manager.finalize_and_merge()
        return "All verified optimizations have been merged to the main project root successfully."
    except Exception as e:
        return f"Error during merge: {str(e)}"

if __name__ == "__main__":
    mcp.run()
